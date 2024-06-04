// vertex parallel: each warp takes one vertex

#include "graph_nvshmem.h"
// #include "cross_gpu_fifo.h"
#include "nvshmem_buffer.h"
// #include "shared_memory_buffer.h"


#define NUM_SENDER_WARPS    456
#define NUM_RECVER_WARPS    456


__device__ AccType worker_launch(GraphNVSHMEM g, int stride) {
    AccType count = 0;

    // about the threads and blocks
    int thread_lane = threadIdx.x & (WARP_SIZE-1);          // thread index within the warp
    int warp_lane = threadIdx.x / WARP_SIZE;                // warp index within the CTA
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;  // global thread index
    int warp_id = thread_id / WARP_SIZE;                    // global warp index

    int worker_warp_id = warp_id - NUM_SENDER_WARPS - NUM_RECVER_WARPS;

    // begin calculation
    vidType n_real_vertices = g.get_n_real_vertices();
    int u_partition_num = nvshmem_my_pe();
    for (int u_local_id = worker_warp_id; u_local_id < n_real_vertices; u_local_id += stride) {
        vidType u = g.get_vertex_in_vertex_list(u_local_id);

        eidType u_list_begin = g.edge_begin(u);
        eidType u_list_end = g.edge_end(u);
        vidType u_degree = u_list_end - u_list_begin;

        for (eidType v_id_in_u_list = u_list_begin; v_id_in_u_list < u_list_end; v_id_in_u_list++) {
            vidType v = g.getEdgeDst(v_id_in_u_list);
            int v_partition_num = g.get_vertex_partition_number(v);
            if (v_partition_num == u_partition_num) {
                count += intersect_num(g.N(u), u_degree, g.N(v), g.get_degree(v));
            }
        }
    }
    return count;
}


__device__ AccType sender_launch(GraphNVSHMEM g, NVSHMEMBuffer push_buffer) {
    AccType count = 0;

    // about the threads and blocks
    int thread_lane = threadIdx.x & (WARP_SIZE-1);          // thread index within the warp
    int warp_lane = threadIdx.x / WARP_SIZE;                // warp index within the CTA
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;  // global thread index
    int warp_id = thread_id / WARP_SIZE;                    // global warp index

    // which destination am I going to take care of
    int ndevices = nvshmem_n_pes();
    int dest_warp_group_size = NUM_SENDER_WARPS / (ndevices - 1);
    int u_partition_num = nvshmem_my_pe();

    int warp_id_in_sender_warps = warp_id;
    if (warp_id_in_sender_warps >= dest_warp_group_size * (ndevices - 1)) return 0;   // boundary warps
    int offset = warp_id_in_sender_warps / dest_warp_group_size + 1;
    int dest_id = (offset + nvshmem_my_pe()) % ndevices;
    if (thread_lane == 0) {
        push_buffer.producer_init_dest_pe(warp_id_in_sender_warps, dest_id);
    } __syncwarp();

    // get my pull buffer
    vidType *my_pull_buffer = g.get_my_pull_buffer(warp_id);
    vidType *my_push_buffer = g.get_my_push_buffer(warp_id);

    // begin scanning
    vidType n_real_vertices = g.get_n_real_vertices();
    for (int u_local_id = warp_id_in_sender_warps % dest_warp_group_size; u_local_id < n_real_vertices; u_local_id += dest_warp_group_size) {
        // if (warp_id_in_sender_warps == 0 && nvshmem_my_pe() == 0 && warp_id_in_sender_warps == 0) {
        //     printf("loop start, thread_lane = %d, u_local_id = %d, n_real_vertices = %d\n", thread_lane, u_local_id, n_real_vertices);
        // } __syncwarp();

        vidType u = g.get_vertex_in_vertex_list(u_local_id);

        eidType u_list_begin = g.edge_begin(u);
        eidType u_list_end = g.edge_end(u);
        vidType u_degree = u_list_end - u_list_begin;

        // initialize the push buffer
        // format:
        // | type | u | u_degree | num_push_tasks | u_list | tasks |
        if (thread_lane == 0) { // TODO: make the copy cleverer
            my_push_buffer[0] = 0;  // normal message
            my_push_buffer[1] = u;
            my_push_buffer[2] = u_degree;
        }
        __syncwarp();   // TODO: can we remove it?

        // make the update to the buffer easier
        vidType *push_u_list = my_push_buffer + 4;
        vidType *push_task_list = my_push_buffer + 4 + u_degree;
        
        // scan the v
        int num_push_tasks = 0;
        for (vidType v_id_in_u_list = 0; v_id_in_u_list < u_degree; v_id_in_u_list++) {
            vidType v = (g.N(u))[v_id_in_u_list];

            // write v to the push buffer
            if (thread_lane == 0) {
                push_u_list[v_id_in_u_list] = v;
            }
            __syncwarp();

            int v_partition_num = g.get_vertex_partition_number(v);
            if (v_partition_num != dest_id) continue;

            // TODO: improve the way the data is organized, such that v's degree can be easily got
            eidType v_list_begin = nvshmem_int64_g(g.get_rowptr()+v, v_partition_num);
            eidType v_list_end = nvshmem_int64_g(g.get_rowptr()+v+1, v_partition_num);
            vidType v_degree = v_list_end - v_list_begin;

            // if (thread_lane == 0) {
            //     push_task_list[num_push_tasks] = v;
            // }
            // __syncwarp();   // TODO: can we remove it?
            // num_push_tasks++;

            // nvshmemx_int32_get_warp(my_pull_buffer, g.get_colidx()+v_list_begin, v_degree, dest_id);
            // count += intersect_num(g.N(u), u_degree, my_pull_buffer, v_degree);

            if (u_degree < v_degree) {  // push
                if (thread_lane == 0) {
                    push_task_list[num_push_tasks] = v;
                }
                __syncwarp();   // TODO: can we remove it?
                num_push_tasks++;
            } else {    // pull
                nvshmemx_int32_get_warp(my_pull_buffer, g.get_colidx()+v_list_begin, v_degree, dest_id);
                count += intersect_num(g.N(u), u_degree, my_pull_buffer, v_degree);
            }
        }

        // if (warp_id_in_sender_warps == 0 && nvshmem_my_pe() == 0 && warp_id_in_sender_warps == 0) {
        //     printf("loop mid, thread_lane = %d, u_local_id = %d, n_real_vertices = %d\n", thread_lane, u_local_id, n_real_vertices);
        // }
        // __syncwarp();

        // send out the message from u
        if (num_push_tasks > 0) {
            if (thread_lane == 0) {
                my_push_buffer[3] = num_push_tasks;
            }
            __syncwarp();   // TODO: can we remove it?

            // if (thread_lane == 0 && nvshmem_my_pe() == 0 && warp_id_in_sender_warps == 0) {
            //     printf("producer_warp_write_msg(%d, my_push_buffer, %d)\n", warp_id_in_sender_warps, 4 + u_degree + num_push_tasks);
            // } __syncwarp();

            push_buffer.producer_warp_write_msg(warp_id_in_sender_warps, my_push_buffer, 4 + u_degree + num_push_tasks);
        }

        // if (warp_id_in_sender_warps == 0 && nvshmem_my_pe() == 0 && warp_id_in_sender_warps == 0) {
        //     printf("loop end, thread_lane = %d, u_local_id = %d, n_real_vertices = %d\n", thread_lane, u_local_id, n_real_vertices);
        // } __syncwarp();
    }

    if (thread_lane == 0) {
        my_push_buffer[0] = 1;  // exit
    } __syncwarp();
    // __threadfence();

    // if (thread_lane == 0 && nvshmem_my_pe() == 0 && warp_id_in_sender_warps == 0) {
    //     printf("producer_warp_write_msg(%d, my_push_buffer, 1)\n", warp_id_in_sender_warps);
    // } __syncwarp();

    // if (thread_lane == 0 && nvshmem_my_pe() == 0 && warp_id_in_sender_warps == 0) {
    //     printf("producer_warp_write_msg--(%d, my_push_buffer, 1)\n", warp_id_in_sender_warps);
    // }
    // __syncwarp();

    push_buffer.producer_warp_write_msg(warp_id_in_sender_warps, my_push_buffer, 1);

    // if (thread_lane == 0 && nvshmem_my_pe() == 0 && warp_id_in_sender_warps == 0) {
    //     printf("pe 0 warp 0 exit\n");
    // } __syncwarp();

    // if (thread_lane == 0) {
    //     printf("pe [%d->%d] sender warp [%d] exit.\n", nvshmem_my_pe(), dest_id, warp_id_in_sender_warps);
    // } __syncwarp();

    // printf("pe[%d] sender[%d] thread[%d] count = %d\n", nvshmem_my_pe(), warp_id_in_sender_warps, thread_lane, count);
    return count;
}


__device__ AccType recver_launch(GraphNVSHMEM g, NVSHMEMBuffer push_buffer, int *lock) {
    AccType count = 0;

    int thread_lane = threadIdx.x & (WARP_SIZE-1);          // thread index within the warp
    int warp_lane = threadIdx.x / WARP_SIZE;                // warp index within the CTA
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;  // global thread index
    int warp_id = thread_id / WARP_SIZE;                    // global warp index

    int warp_id_in_recver_warps = warp_id - NUM_SENDER_WARPS;
    int ndevices = nvshmem_n_pes();
    int dest_warp_group_size = NUM_RECVER_WARPS / (ndevices - 1);
    if (warp_id_in_recver_warps >= dest_warp_group_size * (ndevices - 1)) return 0;   // boundary warps
    int pe_offset = warp_id_in_recver_warps / dest_warp_group_size + 1;
    int src_pe_id = (nvshmem_my_pe() + ndevices - pe_offset) % ndevices;
    if (thread_lane == 0) {
        push_buffer.consumer_init_source_pe(warp_id_in_recver_warps, src_pe_id);
    } __syncwarp();


    while (1) {
        vidType *msg = push_buffer.consumer_warp_get_msg(warp_id_in_recver_warps);
        if (msg[0] == 1) {  // it's time to stop
            // if (thread_lane == 0) {
            //     printf("pe [%d<-%d] recver warp [%d] exit.\n", nvshmem_my_pe(), src_pe_id, warp_id_in_recver_warps);
            // } __syncwarp();
            // printf("pe[%d] recver[%d] thread[%d] count = %d\n", nvshmem_my_pe(), warp_id_in_recver_warps, thread_lane, count);
            return count;
        }

        // decode the received message and do set intersection
        // format:
        // | type | u | u_degree | num_push_tasks | u_list | tasks |
        vidType u_degree = msg[2];
        vidType num_push_tasks = msg[3];
        vidType *u_list = &msg[4];
        vidType *tasks_list = &msg[4 + u_degree];
        for (int i = 0; i < num_push_tasks; i++) {
            vidType v = tasks_list[i];
            vidType v_degree = g.get_degree(v);
            count += intersect_num(u_list, u_degree, g.N(v), v_degree);
        }

        // if (thread_lane == 0) {
        //     while (nvshmem_int_atomic_swap(lock, 1, 0))
        //         ;
        //     printf("pe[%d<-%d] received %d:", nvshmem_my_pe(), src_pe_id, msg[1]);
        //     for (int i = 0; i < num_push_tasks; i++) {
        //         printf(" %d", tasks_list[i]);
        //     }
        //     printf("\n");
        //     nvshmem_int_atomic_swap(lock, 0, 0);
        // }
        // __syncwarp();

        push_buffer.consumer_warp_release_msg(warp_id_in_recver_warps);
    }
}


__global__ void warp_vertex_nvshmem(AccType *total, GraphNVSHMEM g, NVSHMEMBuffer push_buffer, int *lock) {
    __shared__ typename BlockReduce::TempStorage temp_storage;

    // int num_sender_recver_blocks = gridDim.x / 9;   // TODO: change it back to normal
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;  // global thread index
    int warp_id = thread_id / WARP_SIZE;                    // global warp index
    int total_num_of_warps = (gridDim.x * blockDim.x) / WARP_SIZE;

    AccType count = 0;
    if (warp_id < NUM_SENDER_WARPS) {
        count = sender_launch(g, push_buffer);
        // count = 0;
    } else if (warp_id < NUM_SENDER_WARPS + NUM_RECVER_WARPS) {
        count = recver_launch(g, push_buffer, lock);
        // count = 0;
    } else {
        count = worker_launch(g, total_num_of_warps - NUM_SENDER_WARPS - NUM_RECVER_WARPS);
        // count = 0;
    }

    // if (warp_id < total_num_of_warps - 6) {
    //     count = worker_launch(g, total_num_of_warps - 6);
    // } else if (warp_id < ) {
    //     count = sender_launch(g, push_buffer, gridDim.x - 2 * num_sender_recver_blocks, num_sender_recver_blocks);
    // } else {
    //     count = recver_launch(g, push_buffer, gridDim.x - num_sender_recver_blocks, num_sender_recver_blocks);
    // }

    // reduce
    AccType block_num = BlockReduce(temp_storage).Sum(count);
    if (threadIdx.x == 0) atomicAdd(total, block_num);
}























































// __device__ AccType worker_launch(shared_memory_buffer_t *smbuffer, GraphNVSHMEM g) {
//     AccType count = 0;

//     // about the threads and blocks
//     int thread_lane = threadIdx.x & (WARP_SIZE-1);          // thread index within the warp
//     int warp_lane = threadIdx.x / WARP_SIZE;                // warp index within the CTA
//     int thread_id = blockIdx.x * blockDim.x + threadIdx.x;  // global thread index
//     int warp_id = thread_id / WARP_SIZE;                    // global warp index

//     // my warp id among all the worker warps
//     int worker_warp_id = blockIdx.x * 7 + warp_lane;    // TODO: change it back to normal

//     // begin calculation
//     vidType n_real_vertices = g.get_n_real_vertices();
//     int n_total_worker_warps_in_a_grid = gridDim.x / 9 * 8 * 7; // TODO: change it back to normal
//     int u_partition_num = nvshmem_my_pe();
//     for (int u_local_id = worker_warp_id; u_local_id < n_real_vertices; u_local_id += n_total_worker_warps_in_a_grid) {
//         vidType u = g.get_vertex_in_vertex_list(u_local_id);
//         assert(u_partition_num == g.get_vertex_partition_number(u));    // TODO: remove it because it hurts performance

//         eidType u_list_begin = g.edge_begin(u);
//         eidType u_list_end = g.edge_end(u);
//         vidType u_degree = u_list_end - u_list_begin;

//         // copy u's list to the msg_buffer
//         // TODO: create a new API to let all the threads participate in the copy
//         // TODO: try first scan, then compute

//         // format:
//         // | u | u_degree | u_list | num_push_tasks | tasks |
//         producer_wait_warp(smbuffer, warp_lane, nvshmem_my_pe(), nvshmem_n_pes());
//         producer_load_source_vertex_info_warp(smbuffer, u, u_degree, g.N(u), warp_lane, nvshmem_my_pe(), nvshmem_n_pes());
//         int num_push_tasks = 0;
//         for (eidType v_id_in_u_list = u_list_begin; v_id_in_u_list < u_list_end; v_id_in_u_list++) {
//             vidType v = g.getEdgeDst(v_id_in_u_list);   // TODO: can we use the data in the shared memory buffer?
//             int v_partition_num = g.get_vertex_partition_number(v);
//             if (v_partition_num == u_partition_num) {   // local
//                 count += intersect_num(g.N(u), u_degree, g.N(v), g.get_degree(v));
//             } else {    // not local
//                 // TODO: improve the way data is stored, such that v's degree can be easily got
//                 eidType v_list_begin = nvshmem_int64_g(g.get_rowptr()+v, v_partition_num);
//                 eidType v_list_end = nvshmem_int64_g(g.get_rowptr()+v+1, v_partition_num);
//                 vidType v_degree = v_list_end - v_list_begin;
//                 assert(v_degree >= 0);  // TODO: remove it. it is stupid.

//                 if (u_degree < v_degree) {  // push
//                     producer_write_warp(smbuffer, v_partition_num, warp_lane, 1+1+u_degree+1+num_push_tasks, v, nvshmem_my_pe());
//                 } else {    // pull
//                     // TODO: make it also a block shared memory, maybe (but the block shared memory is too small)
//                     vidType *my_pull_buffer = g.get_my_pull_buffer(warp_id);
//                     nvshmemx_int32_get_warp(my_pull_buffer, g.get_colidx()+v_list_begin, v_degree, v_partition_num);
//                     count += intersect_num(g.N(u), u_degree, my_pull_buffer, v_degree);
//                 }
//             }
//         }

//         // TODO: make it clever, issue at a time
//         for (int dest_id = 0; dest_id < nvshmem_n_pes(); dest_id++) {
//             if (dest_id == nvshmem_my_pe()) continue;
//             producer_issue_warp(smbuffer, nvshmem_my_pe(), dest_id, warp_lane);
//         }
//     }

//     producer_exit_warp(smbuffer);
//     if (thread_lane == 0) {
//         printf("warp %d exits.\n", warp_lane);
//     }

//     return count;
// }


// __device__ void sender_launch(shared_memory_buffer_t *smbuffer, NVSHMEMBuffer push_buffer) {
//     int thread_lane = threadIdx.x & (WARP_SIZE-1);

//     while (1) {
//         vidType *msg_head;
//         int msg_length;
//         int dest_id;
//         int from_warp;
//         if (consumer_get_msg_warp(smbuffer, &msg_head, &msg_length, &dest_id, &from_warp, nvshmem_my_pe(), nvshmem_n_pes()) == 1) { // it's time to stop
//             break;
//         }
//         push_buffer.producer_write_msg(blockIdx.x, msg_length, msg_head, dest_id, 0);
//     }

//     // TODO: make it clever, complete for one api call
//     for (int dest_id = 0; dest_id < nvshmem_n_pes(); dest_id++) {
//         if (dest_id == nvshmem_my_pe()) continue;
//         push_buffer.producer_write_msg(blockIdx.x, 0, 0, dest_id, 1);
//     }
// }


// __device__ AccType recver_launch(NVSHMEMBuffer push_buffer, GraphNVSHMEM g) {
//     AccType count = 0;
    
//     int thread_lane = threadIdx.x & (WARP_SIZE-1);          // thread index within the warp
//     int warp_lane = threadIdx.x / WARP_SIZE;                // warp index within the CTA
//     int thread_id = blockIdx.x * blockDim.x + threadIdx.x;  // global thread index
//     int warp_id = thread_id / WARP_SIZE;                    // global warp index

//     int recver_warp_id = warp_id - (warp_id / 9 * 8);
    
//     vidType *start;
//     int degree;
//     int dest;
//     while (1) {
//         if (push_buffer.consumer_get_msg_pointer(recver_warp_id, &start, &degree, &dest) == 1) { // it's time to stop
//             return;
//         }
        
//         // decode the received message and do set intersection
//         // format:
//         // | u | u_degree | u_list | num_push_tasks | tasks |
//         vidType u_degree = start[1];
//         vidType *u_list = &start[2];
//         vidType num_push_tasks = start[2+u_degree];
//         vidType *tasks_list = &start[2+u_degree+1];
//         for (int i = 0; i < num_push_tasks; i++) {
//             vidType v = tasks_list[i];
//             vidType v_degree = g.get_degree(v);
//             count += intersect_num(u_list, u_degree, g.N(v), v_degree);
//         }

//         push_buffer.consumer_release(recver_warp_id, dest);
//     }

//     return count;
// }


// __global__ void warp_vertex_nvshmem(AccType *total, GraphNVSHMEM g, NVSHMEMBuffer push_buffer) {
//     __shared__ typename BlockReduce::TempStorage temp_storage;
    
//     // about the threads and blocks
//     int thread_lane = threadIdx.x & (WARP_SIZE-1);
//     int warp_lane = threadIdx.x / WARP_SIZE;

//     // allocate block shared memory, typically for workers and senders
//     // TODO: can we branch the recvers out before allocating this?
//     __shared__ shared_memory_buffer_t smbuffer;

//     __syncthreads();    // TODO: can we remove it?
//     init_smbuffer_block(&smbuffer);

//     AccType count = 0;
//     if (blockIdx.x < gridDim.x / 9 * 8) {   // TODO: change it back to normal
//         if (warp_lane < NUM_WARPS_IN_A_BLOCK - 1) { // worker
//             count = worker_launch(&smbuffer, g);
//         } else {    // sender
//             sender_launch(&smbuffer, push_buffer);
//         }
//     } else {    // recver
//         count = recver_launch(push_buffer, g);
//     }

//     // reduce
//     AccType block_num = BlockReduce(temp_storage).Sum(count);
//     if (threadIdx.x == 0) atomicAdd(total, block_num);
// }










































































// __device__ void worker_launch(int *valid_for_sender, int *num_working_workers, int *msg_len, vidType *msg_buffer, vidType *sum_test) {
//     AccType count = 0;
    
//     // about the threads and blocks
//     int thread_lane = threadIdx.x & (WARP_SIZE-1);          // thread index within the warp
//     int warp_lane = threadIdx.x / WARP_SIZE;                // warp index within the CTA
//     int thread_id = blockIdx.x * blockDim.x + threadIdx.x;  // global thread index
//     int warp_id = thread_id / WARP_SIZE;                    // global warp index

//     // my id among all the worker warps
//     int worker_warp_id = blockIdx.x * 7 + warp_lane;    // TODO: change it to the general form

//     // begin calculation
//     vidType n_real_vertices = g.get_n_real_vertices();
//     int n_total_worker_warps_in_a_grid = gridDim.x / 9 * 7; // TODO: change it to the general form
//     int u_partition_num = nvshmem_my_pe();
//     for (int u_local_id = worker_warp_id; u_local_id < n_real_vertices; u_local_id += n_total_worker_warps_in_a_grid) {
//         vidType u = g.get_vertex_in_vertex_list(u_local_id);
//         assert(u_partition_num == g.get_vertex_partition_number(u));    // TODO: remove it because it hurts performance

//         eidType u_list_begin = g.edge_begin(u);
//         eidType u_list_end = g.edge_end(u);
//         vidType u_degree = u_list_end - u_list_begin;

//         // copy u's list to the msg_buffer
//         // TODO: let all the threads in the warp participate in the copy
//         if (thread_lane == 0) {
//             // format:
//             // | u | u_degree | u_list | num_push_tasks | tasks |
//             msg_buffer[0] = u;
//             msg_buffer[1] = u_degree;
//             for (int i = 0; i < u_degree; i++) {
//                 eidType v_id_in_u_list = u_list_begin + i;
//                 vidType v = g.getEdgeDst(v_id_in_u_list);
//                 msg_buffer[i+2] = v;
//             }
//             __threadfence_block();  // TODO: can we remove it?
//         }
//         __syncwarp();   // TODO: can we remove it?

//         int num_push_tasks = 0;
//         for (int i = 2; i < u_degree+2; i++) {
//             vidType v = msg_buffer[i];
//             int v_partition_num = g.get_vertex_partition_number(v);
//             if (v_partition_num == u_partition_num) {   // local
//                 count += intersect_num(g.N(u), u_degree, g.N(v), g.get_degree(v));
//             } else {    // not local
//                 // TODO: improve the way data is stored, such that v's degree can be easily got
//                 eidType v_list_begin = nvshmem_int64_g(g.get_rowptr()+v, v_partition_num);
//                 eidType v_list_end = nvshmem_int64_g(g.get_rowptr()+v+1, v_partition_num);
//                 vidType v_degree = v_list_end - v_list_begin;
//                 assert(v_degree >= 0);  // TODO: remove it. it is stupid.

//                 if (u_degree < v_degree) {  // push
//                     if (thread_lane == 0) {
//                         msg_buffer[2 + u_degree + 1 + num_push_tasks] = v;
//                         num_push_tasks++;
//                     }
//                     __syncwarp();
//                 } else {    // pull
//                     // TODO: make it also a block shared memory
//                     vidType *my_pull_buffer = g.get_my_pull_buffer(warp_id);
//                     nvshmemx_int32_get_warp(my_pull_buffer, g.get_colidx()+v_list_begin, v_degree, v_partition_num);
//                     count += intersect_num(g.N(u), u_degree, my_pull_buffer, v_degree);
//                 }
//             }
//         }
//     }
// }

// __device__ void sender_launch(int *valid_for_sender, int *num_working_workers, int *msg_len, vidType *msg_buffer, vidType *sum_test, NVSHMEMBuffer push_buffer) {
//     int thread_lane = threadIdx.x & (WARP_SIZE-1);

//     while (1) {
//         int done = 0;
//         __threadfence_block();
//         if (*num_working_workers == 0) {
//             done = 1;
//         }

//         for (int worker_id = 0; worker_id < NUM_WARPS_IN_A_BLOCK; worker_id++) {
//             __threadfence_block();
//             if (valid_for_sender[worker_id] == 1) {
//                 int len = msg_len[worker_id];
//                 push_buffer.producer_write_msg(blockIdx.x, len, &msg_buffer[worker_id * PRE_ASSUMED_MAX_DEGREE], )
//             }
//         }
//     }
// }

// __device__ void recver_launch() {
    
// }

// __global__ void warp_vertex_nvshmem(AccType *total, GraphNVSHMEM g, NVSHMEMBuffer push_buffer) {
//     // about the threads and blocks
//     int thread_lane = threadIdx.x & (WARP_SIZE-1);
//     int warp_lane = threadIdx.x / WARP_SIZE;

//     // allocate block shared memory, typically for workers and senders
//     // TODO: can we branch the recvers out before allocating this?
//     __shared__ int valid_for_sender[NUM_WARPS_IN_A_BLOCK - 1];
//     __shared__ int num_working_workers;
//     __shared__ int msg_len[NUM_WARPS_IN_A_BLOCK - 1];
//     __shared__ vidType msg_buffer[NUM_WARPS_IN_A_BLOCK * PRE_ASSUMED_MAX_DEGREE];   // TODO: there should be less
//     __shared__ vidType sum_test;    // TODO: remove it

//     __syncthreads();    // TODO: can we remove it?

//     if (warp_lane == 0 && thread_lane == 0) {
//         assert(NUM_WARPS_IN_A_BLOCK == 8);
//         num_working_workers = NUM_WARPS_IN_A_BLOCK - 1;
//         sum_test = 0;
//     }

//     if (warp_lane < NUM_WARPS_IN_A_BLOCK - 1 && thread_lane == 0) {
//         valid_for_sender[warp_lane] = 0;
//     }
//     __threadfence_block();

//     __syncthreads();

//     if (blockIdx.x < gridDim.x / 9 * 8) {   // worker or sender
//         if (warp_lane < NUM_WARPS_IN_A_BLOCK - 1) { // worker
//             worker_launch(&valid_for_sender[warp_lane], &num_working_workers, &msg_len[warp_lane], &msg_buffer[warp_lane * PRE_ASSUMED_MAX_DEGREE], &sum_test);
//         } else {    // sender
//             sender_launch(valid_for_sender, &num_working_workers, msg_len, msg_buffer, &sum_test, push_buffer);
//         }
//     } else {    // recver
//         recver_launch();
//     }
// }


























































// #define RECVER_RATIO    10      // 1/10
// #define SENDER_RATIO    20      // 1/20

// // #define ENTER_KERNEL

// __device__ AccType normal_launch(GraphNVSHMEM g, CrossGPUFIFO fifo, int n_normal_blocks) {
//     AccType count = 0;
    
//     // about the threads and blocks
//     int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;    // global thread index
//     int warp_id = thread_id / WARP_SIZE;                        // global warp index
//     int num_warps = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;       // total number of global warps
//     int thread_lane = threadIdx.x & (WARP_SIZE-1);              // thread index within the warp
//     int warp_lane = threadIdx.x / WARP_SIZE;                    // warp index within the CTA

//     vidType n_real_vertices = g.get_n_real_vertices();
//     int n_total_normal_warps_in_a_grid = n_normal_blocks * blockDim.x / WARP_SIZE;
//     int u_partition_num = nvshmem_my_pe();
//     for (int u_local_id = warp_id; u_local_id < n_real_vertices; u_local_id += n_total_normal_warps_in_a_grid) {
//         vidType u = g.get_vertex_in_vertex_list(u_local_id);
//         assert(u_partition_num == g.get_vertex_partition_number(u));
        
//         eidType u_list_begin = g.edge_begin(u);
//         eidType u_list_end = g.edge_end(u);
//         vidType u_degree = u_list_end - u_list_begin;
//         for (eidType v_id_in_u_list = u_list_begin; v_id_in_u_list < u_list_end; v_id_in_u_list++) {
//             vidType v = g.getEdgeDst(v_id_in_u_list);

//             int v_partition_num = g.get_vertex_partition_number(v);
//             if (v_partition_num == u_partition_num) {   // local
//                 count += intersect_num(g.N(u), u_degree, g.N(v), g.get_degree(v));
//             } else {    // not local
//                 eidType v_list_begin = nvshmem_int64_g(g.get_rowptr()+v, v_partition_num);  // these can be further optimized
//                 eidType v_list_end = nvshmem_int64_g(g.get_rowptr()+v+1, v_partition_num);
//                 vidType v_degree = v_list_end - v_list_begin;
//                 assert(v_degree >= 0);

//                 if (u_degree < v_degree) {  // push
//                     fifo.local_fifo_update(0, u, v, v_partition_num, warp_id);
//                 } else {    // pull
//                     vidType *my_pull_buffer = g.get_my_pull_buffer(warp_id);
//                     nvshmemx_int32_get_warp(my_pull_buffer, g.get_colidx()+v_list_begin, v_degree, v_partition_num);
//                     count += intersect_num(g.N(u), u_degree, my_pull_buffer, v_degree);
//                 }
//             }
//         }
//         fifo.local_fifo_update_finished(u, warp_id);
//     }
//     fifo.local_fifo_update(1, 0, 0, 0, warp_id);    // this normal warp has finished updating
//     return count;
// }

// __device__ void sender_launch(GraphNVSHMEM g, CrossGPUFIFO fifo, int n_normal_blocks) {
//     // about the threads and blocks
//     int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;    // global thread index
//     int warp_id = thread_id / WARP_SIZE;                        // global warp index
//     int num_warps = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;       // total number of global warps
//     int thread_lane = threadIdx.x & (WARP_SIZE-1);              // thread index within the warp
//     int warp_lane = threadIdx.x / WARP_SIZE;                    // warp index within the CTA

//     fifo.check_local(g, warp_id, n_normal_blocks);
// }

// __device__ AccType recver_launch(GraphNVSHMEM g, CrossGPUFIFO fifo) {
//     AccType count = 0;

//     // about the threads and blocks
//     int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;    // global thread index
//     int warp_id = thread_id / WARP_SIZE;                        // global warp index
//     int num_warps = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;       // total number of global warps
//     int thread_lane = threadIdx.x & (WARP_SIZE-1);              // thread index within the warp
//     int warp_lane = threadIdx.x / WARP_SIZE;                    // warp index within the CTA
    
//     vidType u;
//     vidType task_count;
//     vidType *tasks;
//     vidType degree;
//     vidType *list;

//     while (1) {
//         int channel;
//         int buffer_id;
//         if (fifo.recv(warp_id, &u, &task_count, &tasks, &degree, &list, &channel, &buffer_id) == 1) {
//             break;
//         }
//         for (vidType task_id = 0; task_id < task_count; task_id++) {
//             vidType v = tasks[task_id];
//             count += intersect_num(list, degree, g.N(v), g.get_degree(v));
//         }
//         fifo.release_message_buffer(channel, buffer_id);
//     }
//     return count;
// }

// __global__ void warp_vertex_nvshmem(AccType *total, GraphNVSHMEM g, CrossGPUFIFO fifo, int ndevices) {
//     __shared__ typename BlockReduce::TempStorage temp_storage;

// #ifdef ENTER_KERNEL
//     if (threadIdx.x == 0 && blockIdx.x == 0) {
//         printf("pe %d entered.\n", nvshmem_my_pe());
//     }
// #endif
    
//     // allocate tasks
//     int n_blocks = gridDim.x;
//     assert(n_blocks >= SENDER_RATIO);
//     int n_recver_blocks = n_blocks / RECVER_RATIO;
//     int n_sender_blocks = n_blocks / SENDER_RATIO;
//     assert(n_sender_blocks * (BLOCK_SIZE / WARP_SIZE) >= (ndevices-1) * NUM_CHANNELS);
//     int n_normal_blocks = n_blocks - n_recver_blocks - n_sender_blocks;

//     // senders, receivers and normals go to different branches
//     AccType count = 0;
//     if (blockIdx.x < n_normal_blocks) {
//         count = normal_launch(g, fifo, n_normal_blocks);
//     } else if (blockIdx.x < n_normal_blocks + n_sender_blocks) {
//         sender_launch(g, fifo, n_normal_blocks);
//     } else {
//         count = recver_launch(g, fifo);
//     }

//     // reduce
//     AccType block_num = BlockReduce(temp_storage).Sum(count);
//     if (threadIdx.x == 0) atomicAdd(total, block_num);
// }
