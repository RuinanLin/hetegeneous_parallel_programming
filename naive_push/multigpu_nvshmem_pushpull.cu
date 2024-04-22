#include "graph.h"
#include "graph_partition.h"
#include "nvshmem.h"
#include "nvshmemx.h"
#include "cutil_subset.h"
#include "common.h"
#include "graph_gpu.h"
#include "bs_warp_vertex_nvshmem.cuh"
#include "cuda_launch_config.hpp"

void TCSolver(Graph &g, uint64_t &total, int n_partitions, int chunk_size) {
    auto nv = g.V();                // number of vertices
    auto md = g.get_max_degree();   // max degree
    int ndevices;                   // how many devices
    CUDA_SAFE_CALL(cudaGetDeviceCount(&ndevices));
    if (ndevices < n_partitions) {
        std::cout << "Only " << ndevices << " GPUs available!\n";
        exit(1);
    } else ndevices = n_partitions;
    eidType max_subg_ne = 0;

    // partition the graph using metis and store it into pg
    PartitionedGraph pg(&g, ndevices);
    pg.metis_partition();
    auto num_subgraphs = pg.get_num_subgraphs();
    assert(num_subgraphs == n_partitions);
    for (int i = 0; i < ndevices; i++) {    // get the maximum subgraph ne
        auto subg_ne = pg.get_subgraph(i)->E();
        if (subg_ne > max_subg_ne)
            max_subg_ne = subg_ne;
    }

    // initialization for nvshmem
    nvshmem_init();
    int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    std::cout << "mype_node = " << mype_node << "\n";
    CUDA_SAFE_CALL(cudaSetDevice(mype_node));
    cudaStream_t stream;
    CUDA_SAFE_CALL(cudaStreamCreate(&stream));

    int npes = nvshmem_n_pes();
    int mype = nvshmem_my_pe();
    auto &subg = *pg.get_subgraph(mype);

    // calculate the numbers about the gpus
    size_t nthreads = BLOCK_SIZE;
    size_t nblocks = 65536;
    cudaDeviceProp deviceProp;
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
    int max_blocks_per_SM = maximum_residency(warp_vertex_nvshmem, nthreads, 0);
    std::cout << "max_blocks_per_SM = " << max_blocks_per_SM << "\n";
    size_t max_blocks = max_blocks_per_SM * deviceProp.multiProcessorCount;
    nblocks = std::min(6*max_blocks, nblocks);
    std::cout << "CUDA triangle counting (" << nblocks << " CTAs, " << nthreads << " threads/CTA)\n";
    size_t nwarps = WARPS_PER_BLOCK;

    // calculation related to the messages
    size_t num_warps_each_gpu_each_iteration = nwarps * nblocks;
    std::cout << "num_warps_each_gpu_each_iteration = " << num_warps_each_gpu_each_iteration << "\n";
    size_t message_valid_size = 4;
    size_t message_u_size = sizeof(vidType);
    size_t message_degree_size = sizeof(vidType);
    size_t message_bitmask_size = ((md-1)/32+1) * 4;
    size_t message_list_size = md * sizeof(vidType);
    size_t single_message_size = message_valid_size + message_u_size + message_degree_size + message_bitmask_size + message_list_size;
    std::cout << "single_message_size = " << single_message_size << "\n";
    size_t message_buffer_size = (ndevices - 1) * num_warps_each_gpu_each_iteration * single_message_size;
    std::cout << "message_buffer_size = " << message_buffer_size << "\n";

    // prepare for the graph on gpu
    Timer t;
    t.Start();
    GraphGPU d_graph;
    d_graph.init_nvshmem(subg, mype, message_buffer_size, stream);
    nvshmemx_barrier_all_on_stream(stream);
    t.Stop();
    std::cout << "PE[" << mype << "] Total time allocating nvshmem and copying subgraphs to GPUs: " << t.Seconds() << " sec\n";

    // launch kernel
    t.Start();
    AccType h_count = 0;
    AccType *d_count = (AccType *)nvshmem_malloc(sizeof(AccType));
    CUDA_SAFE_CALL(cudaMemcpyAsync(d_count, &h_count, sizeof(AccType), cudaMemcpyHostToDevice, stream));
    std::cout << "PE[" << mype << "] Start kernel\n";
    warp_vertex_nvshmem<<<nblocks, nthreads, 0, stream>>>(d_count, d_graph);
    CUDA_SAFE_CALL(cudaMemcpyAsync(&h_count, d_count, sizeof(AccType), cudaMemcpyDeviceToHost, stream));
    t.Stop();
    nvshmemx_barrier_all_on_stream(stream);
    std::cout << "runtime[gpu " << mype << "] = " << t.Seconds() << " sec\n";
    
    nvshmem_finalize();
}