#include "graph.h"
#include "graph_partition.h"
#include "nvshmem.h"
#include "nvshmemx.h"
#include "cutil_subset.h"
#include "common.h"
#include "graph_gpu.h"
#include "cuda_launch_config.hpp"
#include <cub/cub.cuh>
typedef cub::BlockReduce<AccType, BLOCK_SIZE> BlockReduce;
#include "bs_warp_vertex_nvshmem.cuh"

#include <fstream>
#include <string>

#define NVSHMEM_CHECK(stmt)                                                                \
    do {                                                                                   \
        int result = (stmt);                                                               \
        if (NVSHMEMX_SUCCESS != result) {                                                  \
            fprintf(stderr, "[%s:%d] nvshmem failed with error %d \n", __FILE__, __LINE__, \
                    result);                                                               \
            exit(-1);                                                                      \
        }                                                                                  \
    } while (0)

void TCSolver(Graph &g, uint64_t &total, int n_partitions, int chunk_size) {
    auto nv = g.V();                // number of vertices
    auto md = g.get_max_degree();   // max degree
    int ndevices;                   // how many devices
    CUDA_SAFE_CALL(cudaGetDeviceCount(&ndevices));
    if (ndevices < n_partitions) {
        std::cout << "Only " << ndevices << " GPUs available!\n";
        exit(1);
    } else ndevices = n_partitions;
    vidType max_subg_nv = 0;
    eidType max_subg_ne = 0;

    // initialization for nvshmem
    nvshmem_init();
    int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    std::cout << "mype_node = " << mype_node << "\n";
    CUDA_SAFE_CALL(cudaSetDevice(mype_node));

    int npes = nvshmem_n_pes();
    int mype = nvshmem_my_pe();
    nvshmem_barrier_all();

    // // check the input
    // std::string file_name = "input_log.txt";
    // std::ofstream logfile(file_name);
    // if (!logfile) {
    //     std::cerr << "Cannot open " << file_name << "\n";
    //     exit(-1);
    // }
    // if (mype == 0) {
    //     logfile << "rowptr:\n";
    //     for (vidType v = 0; v < g.V()+1; v++) {
    //         logfile << g.edge_begin(v) << " ";
    //     }
    //     logfile << "\n\n";

    //     logfile << "colidx:\n";
    //     for (vidType v = 0; v < g.V(); v++) {
    //         logfile << v << ": ";
    //         for (vidType n = 0; n < g.get_degree(v); n++) {
    //             logfile << g.N(v, n) << " ";
    //         }
    //         logfile << "\n";
    //     }

    //     logfile << "\n";
    // }

    // partition the graph using metis and store it into pg
    PartitionedGraph pg(&g, ndevices);
    pg.metis_partition();
    auto num_subgraphs = pg.get_num_subgraphs();
    assert(num_subgraphs == n_partitions);
    for (int i = 0; i < ndevices; i++) {    // get the maximum subgraph ne
        auto subg_ne = pg.get_subgraph(i)->E();
        if (subg_ne > max_subg_ne)
            max_subg_ne = subg_ne;

        auto subg_nv = pg.get_subgraph(i)->get_n_real_vertices();
        if (subg_nv > max_subg_nv)
            max_subg_nv = subg_nv;
    }

    // // check the partition result
    // if (mype == 0) {
    //     for (int device_id = 0; device_id < ndevices; device_id++) {
    //         logfile << "subgraph " << device_id << "\n";
    //         Graph &subg = *pg.get_subgraph(device_id);
    //         logfile << "rowptr:\n";
    //         for (vidType v = 0; v < subg.V()+1; v++) {
    //             logfile << subg.edge_begin(v) << " ";
    //         }
    //         logfile << "\n";

    //         logfile << "colidx:\n";
    //         for (eidType e = 0; e < subg.E(); e++) {
    //             logfile << subg.getEdgeDst(e) << " ";
    //         }
    //         logfile << "\n\n";
    //     }
    // }

    // calculate the numbers about the gpus
    size_t nthreads = BLOCK_SIZE;
    size_t nblocks = 65536;
    cudaDeviceProp deviceProp;
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
    int max_blocks_per_SM = maximum_residency(warp_vertex_nvshmem_local, nthreads, 0);
    max_blocks_per_SM = std::max(max_blocks_per_SM, (int)maximum_residency(warp_vertex_nvshmem_global, nthreads, 0));
    std::cout << "max_blocks_per_SM = " << max_blocks_per_SM << "\n";
    size_t max_blocks = max_blocks_per_SM * deviceProp.multiProcessorCount;
    nblocks = std::min(max_blocks, nblocks);
    std::cout << "CUDA triangle counting (" << nblocks << " CTAs, " << nthreads << " threads/CTA)\n";
    size_t nwarps = WARPS_PER_BLOCK;
    nvshmem_barrier_all();

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
    int num_iterations = (max_subg_nv-1)/num_warps_each_gpu_each_iteration+1;
    std::cout << "max_subg_nv = " << max_subg_nv << "\n";
    std::cout << "num_iterations = " << num_iterations << "\n"; 
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    nvshmem_barrier_all();

    // prepare for the graph on gpu
    GraphGPU d_graph;
    d_graph.init_nvshmem(pg, mype, message_buffer_size);
    d_graph.set_max_degree(md);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    nvshmem_barrier_all();

    // launch kernel
    Timer t;
    t.Start();
    AccType h_count = 0;
    AccType *d_count = (AccType *)nvshmem_malloc(sizeof(AccType));
    CUDA_SAFE_CALL(cudaMemcpy(d_count, &h_count, sizeof(AccType), cudaMemcpyHostToDevice));
    std::cout << "PE[" << mype << "] Start kernel\n";
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    nvshmem_barrier_all();

    int u_local_id_start = 0;
    int u_local_id_end = num_warps_each_gpu_each_iteration;
    warp_clear_valid<<<nblocks, nthreads>>>(d_graph);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    nvshmem_barrier_all();
    for (int iteration_id = 0; iteration_id < num_iterations; iteration_id++) {
        warp_vertex_nvshmem_local<<<nblocks, nthreads>>>(d_count, d_graph, u_local_id_start, u_local_id_end);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        nvshmem_barrier_all();
        warp_vertex_nvshmem_global<<<nblocks, nthreads>>>(d_count, d_graph);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        nvshmem_barrier_all();
        u_local_id_start += num_warps_each_gpu_each_iteration;
        u_local_id_end += num_warps_each_gpu_each_iteration;
    }

    CUDA_SAFE_CALL(cudaMemcpy(&h_count, d_count, sizeof(AccType), cudaMemcpyDeviceToHost));
    t.Stop();
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    nvshmem_barrier_all();
    std::cout << "runtime[gpu " << mype << "] = " << t.Seconds() << " sec\n";
    total = h_count;
    
    nvshmem_finalize();
    // logfile.close();
}