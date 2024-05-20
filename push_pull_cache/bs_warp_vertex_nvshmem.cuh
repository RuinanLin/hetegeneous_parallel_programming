// vertex parallel: each warp takes one vertex

#include "graph_nvshmem.h"
#include "cross_gpu_fifo.h"

#define RECVER_RATIO    10      // 1/10
#define SENDER_RATIO    20      // 1/20

// #define ENTER_KERNEL

__device__ AccType normal_launch(GraphNVSHMEM g, CrossGPUFIFO fifo, int n_normal_blocks) {
    AccType count = 0;
    
    // about the threads and blocks
    int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;    // global thread index
    int warp_id = thread_id / WARP_SIZE;                        // global warp index
    int num_warps = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;       // total number of global warps
    int thread_lane = threadIdx.x & (WARP_SIZE-1);              // thread index within the warp
    int warp_lane = threadIdx.x / WARP_SIZE;                    // warp index within the CTA

    vidType n_real_vertices = g.get_n_real_vertices();
    int n_total_normal_warps_in_a_grid = n_normal_blocks * blockDim.x / WARP_SIZE;
    int u_partition_num = nvshmem_my_pe();
    for (int u_local_id = warp_id; u_local_id < n_real_vertices; u_local_id += n_total_normal_warps_in_a_grid) {
        vidType u = g.get_vertex_in_vertex_list(u_local_id);
        assert(u_partition_num == g.get_vertex_partition_number(u));
        
        eidType u_list_begin = g.edge_begin(u);
        eidType u_list_end = g.edge_end(u);
        vidType u_degree = u_list_end - u_list_begin;
        for (eidType v_id_in_u_list = u_list_begin; v_id_in_u_list < u_list_end; v_id_in_u_list++) {
            vidType v = g.getEdgeDst(v_id_in_u_list);

            int v_partition_num = g.get_vertex_partition_number(v);
            if (v_partition_num == u_partition_num) {   // local
                count += intersect_num(g.N(u), u_degree, g.N(v), g.get_degree(v));
            } else {    // not local
                eidType v_list_begin = nvshmem_int64_g(g.get_rowptr()+v, v_partition_num);  // these can be further optimized
                eidType v_list_end = nvshmem_int64_g(g.get_rowptr()+v+1, v_partition_num);
                vidType v_degree = v_list_end - v_list_begin;
                assert(v_degree >= 0);

                if (u_degree < v_degree) {  // push
                    fifo.local_fifo_update(0, u, v, v_partition_num, warp_id);
                } else {    // pull
                    vidType *my_pull_buffer = g.get_my_pull_buffer(warp_id);
                    nvshmemx_int32_get_warp(my_pull_buffer, g.get_colidx()+v_list_begin, v_degree, v_partition_num);
                    count += intersect_num(g.N(u), u_degree, my_pull_buffer, v_degree);
                }
            }
        }
        fifo.local_fifo_update_finished(u, warp_id);
    }
    fifo.local_fifo_update(1, 0, 0, 0, warp_id);    // this normal warp has finished updating
    return count;
}

__device__ void sender_launch(GraphNVSHMEM g, CrossGPUFIFO fifo, int n_normal_blocks) {
    // about the threads and blocks
    int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;    // global thread index
    int warp_id = thread_id / WARP_SIZE;                        // global warp index
    int num_warps = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;       // total number of global warps
    int thread_lane = threadIdx.x & (WARP_SIZE-1);              // thread index within the warp
    int warp_lane = threadIdx.x / WARP_SIZE;                    // warp index within the CTA

    fifo.check_local(g, warp_id, n_normal_blocks);
}

__device__ AccType recver_launch(GraphNVSHMEM g, CrossGPUFIFO fifo) {
    AccType count = 0;

    // about the threads and blocks
    int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;    // global thread index
    int warp_id = thread_id / WARP_SIZE;                        // global warp index
    int num_warps = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;       // total number of global warps
    int thread_lane = threadIdx.x & (WARP_SIZE-1);              // thread index within the warp
    int warp_lane = threadIdx.x / WARP_SIZE;                    // warp index within the CTA
    
    vidType u;
    vidType task_count;
    vidType *tasks;
    vidType degree;
    vidType *list;

    while (1) {
        int channel;
        int buffer_id;
        if (fifo.recv(warp_id, &u, &task_count, &tasks, &degree, &list, &channel, &buffer_id) == 1) {
            break;
        }
        for (vidType task_id = 0; task_id < task_count; task_id++) {
            vidType v = tasks[task_id];
            count += intersect_num(list, degree, g.N(v), g.get_degree(v));
        }
        fifo.release_message_buffer(channel, buffer_id);
    }
    return count;
}

__global__ void warp_vertex_nvshmem(AccType *total, GraphNVSHMEM g, CrossGPUFIFO fifo, int ndevices) {
    __shared__ typename BlockReduce::TempStorage temp_storage;

#ifdef ENTER_KERNEL
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("pe %d entered.\n", nvshmem_my_pe());
    }
#endif
    
    // allocate tasks
    int n_blocks = gridDim.x;
    assert(n_blocks >= SENDER_RATIO);
    int n_recver_blocks = n_blocks / RECVER_RATIO;
    int n_sender_blocks = n_blocks / SENDER_RATIO;
    assert(n_sender_blocks * (BLOCK_SIZE / WARP_SIZE) >= (ndevices-1) * NUM_CHANNELS);
    int n_normal_blocks = n_blocks - n_recver_blocks - n_sender_blocks;

    // senders, receivers and normals go to different branches
    AccType count = 0;
    if (blockIdx.x < n_normal_blocks) {
        count = normal_launch(g, fifo, n_normal_blocks);
    } else if (blockIdx.x < n_normal_blocks + n_sender_blocks) {
        sender_launch(g, fifo, n_normal_blocks);
    } else {
        count = recver_launch(g, fifo);
    }

    // reduce
    AccType block_num = BlockReduce(temp_storage).Sum(count);
    if (threadIdx.x == 0) atomicAdd(total, block_num);
}
