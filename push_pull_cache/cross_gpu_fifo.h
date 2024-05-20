#pragma once
#include "graph.h"
#include "nvshmem.h"
#include "nvshmemx.h"
#include <stdlib.h>
#include "graph_nvshmem.h"

#define LOCAL_FIFO_EACH_DEST_LEN        64      // how many messages can be held per destination
#define GLOBAL_FIFO_EACH_CHANNEL_LEN    128
#define NUM_CHANNELS                    4

// #define PRINT_SENDER
// #define PRINT_NORMAL
// #define SENDER_FINISH_SIGNAL
// #define SENDER_START

class CrossGPUFIFO {
private:
    // fifo information
    int fifo_id;
    vidType max_degree;
    int buffer_len;
    int ndevices;
    int normal_warps_finished_counter_h;
    int sender_warps_finished_counter_h;

    // global fifo buffer content
    int *global_occupied;                       // whether this is an empty space for a sender, if so 0, otherwise 1                        [ NUM_CHANNELS, GLOBAL_FIFO_EACH_CHANNEL_LEN ]
    int *global_copy_finished_signal_buffer;    // whether the sender has finished copying its message to this place, if so 1, otherwise 0  [ NUM_CHANNELS, GLOBAL_FIFO_EACH_CHANNEL_LEN ]
    int *global_message_type_buffer;            // 0: normal, 1: finish                                                                     [ NUM_CHANNELS, GLOBAL_FIFO_EACH_CHANNEL_LEN ]
    vidType *global_u_buffer;                   // from which vertex                                                                        [ NUM_CHANNELS, GLOBAL_FIFO_EACH_CHANNEL_LEN ]
    vidType *global_task_count_buffer;          // length of the tasks                                                                      [ NUM_CHANNELS, GLOBAL_FIFO_EACH_CHANNEL_LEN ]
    vidType *global_tasks_buffer;               // content of the tasks                                                                     [ NUM_CHANNELS, GLOBAL_FIFO_EACH_CHANNEL_LEN, max_degree ]
    vidType *global_degree_buffer;              // length of the list                                                                       [ NUM_CHANNELS, GLOBAL_FIFO_EACH_CHANNEL_LEN ]
    vidType *global_list_buffer;                // content of the list                                                                      [ NUM_CHANNELS, GLOBAL_FIFO_EACH_CHANNEL_LEN, max_degree ]

    int *peers_finished_counter;                // how many peers have finished sending                                                     [ 1 ]

    // local fifo buffer content
    int *local_occupied;                        // whether this message area is occupied                                                    [ ndevices-1, LOCAL_FIFO_EACH_DEST_LEN ]
    int *local_normal_warp_updating;            // whether a normal warp is updating tasks in this message                                  [ ndevices-1, LOCAL_FIFO_EACH_DEST_LEN ]
    int *local_ready_for_sender;                // whether a sender warp can grab and process this message                                  [ ndevices-1, LOCAL_FIFO_EACH_DEST_LEN ]
    vidType *local_u_buffer;                    // source vertex u in the local messages                                                    [ ndevices-1, LOCAL_FIFO_EACH_DEST_LEN ]
    vidType *local_task_count_buffer;           // how many tasks in this message                                                           [ ndevices-1, LOCAL_FIFO_EACH_DEST_LEN ]
    vidType *local_tasks_buffer;                // what tasks                                                                               [ ndevices-1, LOCAL_FIFO_EACH_DEST_LEN, max_degree ]

    int *normal_warps_finished_counter;         // how many normal warps are still working                                                  [ 1 ]
    int *sender_warps_finished_counter;         // how many sender warps are still working                                                  [ 1 ]

    // debug helper
    int *print_lock;                            // aquire this lock on GPU0 before any printing job

public:
    CrossGPUFIFO(int mype, int md, size_t n_total_warps, int n_partitions, int n_normal_warps, int n_sender_warps)
     : fifo_id(mype), max_degree(md), buffer_len(n_total_warps), ndevices(n_partitions), normal_warps_finished_counter_h(n_normal_warps), sender_warps_finished_counter_h(n_sender_warps) {        
        // global fifo buffer content
        global_occupied = (int *)nvshmem_malloc(NUM_CHANNELS * GLOBAL_FIFO_EACH_CHANNEL_LEN * sizeof(int));
        global_copy_finished_signal_buffer = (int *)nvshmem_malloc(NUM_CHANNELS * GLOBAL_FIFO_EACH_CHANNEL_LEN * sizeof(int));
        global_message_type_buffer = (int *)nvshmem_malloc(NUM_CHANNELS * GLOBAL_FIFO_EACH_CHANNEL_LEN * sizeof(int));
        global_u_buffer = (vidType *)nvshmem_malloc(NUM_CHANNELS * GLOBAL_FIFO_EACH_CHANNEL_LEN * sizeof(vidType));
        global_task_count_buffer = (vidType *)nvshmem_malloc(NUM_CHANNELS * GLOBAL_FIFO_EACH_CHANNEL_LEN * sizeof(vidType));
        global_tasks_buffer = (vidType *)nvshmem_malloc(NUM_CHANNELS * GLOBAL_FIFO_EACH_CHANNEL_LEN * max_degree * sizeof(vidType));
        global_degree_buffer = (vidType *)nvshmem_malloc(NUM_CHANNELS * GLOBAL_FIFO_EACH_CHANNEL_LEN * sizeof(vidType));
        global_list_buffer = (vidType *)nvshmem_malloc(NUM_CHANNELS * GLOBAL_FIFO_EACH_CHANNEL_LEN * max_degree * sizeof(vidType));
        CUDA_SAFE_CALL(cudaMalloc((void **)&peers_finished_counter, sizeof(int)));
        
        CUDA_SAFE_CALL(cudaMemset(global_occupied, 0, NUM_CHANNELS * GLOBAL_FIFO_EACH_CHANNEL_LEN * sizeof(int)));
        CUDA_SAFE_CALL(cudaMemset(global_copy_finished_signal_buffer, 0, NUM_CHANNELS * GLOBAL_FIFO_EACH_CHANNEL_LEN * sizeof(int)));
        int peers_finished_counter_h = ndevices - 1;
        CUDA_SAFE_CALL(cudaMemcpy(peers_finished_counter, &peers_finished_counter_h, sizeof(int), cudaMemcpyHostToDevice));

        // local fifo buffer content
        CUDA_SAFE_CALL(cudaMalloc((void **)&local_occupied, (ndevices-1) * LOCAL_FIFO_EACH_DEST_LEN * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void **)&local_normal_warp_updating, (ndevices-1) * LOCAL_FIFO_EACH_DEST_LEN * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void **)&local_ready_for_sender, (ndevices-1) * LOCAL_FIFO_EACH_DEST_LEN * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void **)&local_u_buffer, (ndevices-1) * LOCAL_FIFO_EACH_DEST_LEN * sizeof(vidType)));
        CUDA_SAFE_CALL(cudaMalloc((void **)&local_task_count_buffer, (ndevices-1) * LOCAL_FIFO_EACH_DEST_LEN * sizeof(vidType)));
        CUDA_SAFE_CALL(cudaMalloc((void **)&local_tasks_buffer, (ndevices-1) * LOCAL_FIFO_EACH_DEST_LEN * max_degree * sizeof(vidType)));
        CUDA_SAFE_CALL(cudaMalloc((void **)&normal_warps_finished_counter, sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void **)&sender_warps_finished_counter, sizeof(int)));

        CUDA_SAFE_CALL(cudaMemset(local_occupied, 0, (ndevices-1) * LOCAL_FIFO_EACH_DEST_LEN * sizeof(int)));
        CUDA_SAFE_CALL(cudaMemset(local_normal_warp_updating, 0, (ndevices-1) * LOCAL_FIFO_EACH_DEST_LEN * sizeof(int)));
        CUDA_SAFE_CALL(cudaMemset(local_ready_for_sender, 0, (ndevices-1) * LOCAL_FIFO_EACH_DEST_LEN * sizeof(int)));
        CUDA_SAFE_CALL(cudaMemcpy(normal_warps_finished_counter, &normal_warps_finished_counter_h, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(sender_warps_finished_counter, &sender_warps_finished_counter_h, sizeof(int), cudaMemcpyHostToDevice));

        // debug helper
        print_lock = (int *)nvshmem_malloc(sizeof(int));
        CUDA_SAFE_CALL(cudaMemset(print_lock, 0, sizeof(int)));
    }

    inline __device__ int hash(int key, int range) {
        return key % range;
    }

    inline __device__ int local_fifo_find_updating_col(int *local_col_id, int local_row_id, vidType u, int warp_id) { // the return value is the result, found 1, not found 0
        // we start finding from the warp_id's hash value
        int found = 0;
        int find_position_start = hash(warp_id, LOCAL_FIFO_EACH_DEST_LEN);
        int find_position;
        for (int i = 0; i < LOCAL_FIFO_EACH_DEST_LEN; i++) {
            find_position = (find_position_start + i) % LOCAL_FIFO_EACH_DEST_LEN;
            __threadfence();
            if (local_occupied[local_row_id * LOCAL_FIFO_EACH_DEST_LEN + find_position] == 1 && local_normal_warp_updating[local_row_id * LOCAL_FIFO_EACH_DEST_LEN + find_position] == 1 && local_u_buffer[local_row_id * LOCAL_FIFO_EACH_DEST_LEN + find_position] == u) {
                assert(local_ready_for_sender[local_row_id * LOCAL_FIFO_EACH_DEST_LEN + find_position] == 0);
                found = 1;
                break;
            }
        }
        if (found) {
            *local_col_id = find_position;
            return 1;
        }
        return 0;
    }

    inline __device__ int local_fifo_get_not_occupied_col(int local_row_id, int warp_id) {
        // about the thread operating on this function
        int thread_lane = threadIdx.x & (WARP_SIZE-1);
        
        // we start from the warp_id's hash value
        for (int find_position = hash(warp_id, LOCAL_FIFO_EACH_DEST_LEN); ; find_position = (find_position + 1) % LOCAL_FIFO_EACH_DEST_LEN) {
            int aquire_result;
            if (thread_lane == 0) {
                __threadfence();
                aquire_result = atomicExch(&local_occupied[local_row_id * LOCAL_FIFO_EACH_DEST_LEN + find_position], 1);
            }
            aquire_result = __shfl_sync(0xffffffff, aquire_result, 0);
            if (aquire_result == 0) {   // successfully get the lock of the position
                __threadfence();
                assert(local_normal_warp_updating[local_row_id * LOCAL_FIFO_EACH_DEST_LEN + find_position] == 0);
                assert(local_ready_for_sender[local_row_id * LOCAL_FIFO_EACH_DEST_LEN + find_position] == 0);
                return find_position;
            }
        }
    }

    inline __device__ void local_fifo_update(int msg_type, vidType u, vidType v, int dest, int warp_id) {
        // about the thread operating on this function
        int thread_lane = threadIdx.x & (WARP_SIZE-1);
        
        if (msg_type == 1) {    // this warp has finished sending
            if (thread_lane == 0) {
                int old = atomicSub(normal_warps_finished_counter, 1);
                __threadfence();

#ifdef PRINT_NORMAL
                nvshmem_fence();
                while (nvshmem_int_atomic_swap(print_lock, 1, 0) == 1)
                    ;
                printf("pe %d sender warp %d modified normal_warps_finished_counter, old value %d\n", fifo_id, warp_id, old);
                nvshmem_int_atomic_swap(print_lock, 0, 0);
#endif

            }
        } else {    // a normal message
            // get row id
            assert(dest != fifo_id);
            int local_row_id = (dest < fifo_id) ? dest : dest-1;

            // get col id
            int local_col_id;   // [ local_row_id, local_col_id ] is the position to write to
            if (local_fifo_find_updating_col(&local_col_id, local_row_id, u, warp_id) == 0) {  // not found, need to grab a element in the empty list
                local_col_id = local_fifo_get_not_occupied_col(local_row_id, warp_id);
                if (thread_lane == 0) {
                    local_u_buffer[local_row_id * LOCAL_FIFO_EACH_DEST_LEN + local_col_id] = u;
                    local_task_count_buffer[local_row_id * LOCAL_FIFO_EACH_DEST_LEN + local_col_id] = 0;
                    __threadfence();
                    local_normal_warp_updating[local_row_id * LOCAL_FIFO_EACH_DEST_LEN + local_col_id] = 1;
                    __threadfence();
                }
            }

            // insert new v
            if (thread_lane == 0) {
                __threadfence();
                int v_position = local_task_count_buffer[local_row_id * LOCAL_FIFO_EACH_DEST_LEN + local_col_id];
                assert(v_position < max_degree);
                local_tasks_buffer[local_row_id * LOCAL_FIFO_EACH_DEST_LEN * max_degree + local_col_id * max_degree + v_position] = v;
                local_task_count_buffer[local_row_id * LOCAL_FIFO_EACH_DEST_LEN + local_col_id]++;
                __threadfence();
            }
        }
    }

    // source vertex u has finished its message update and tells the sender that it can be sent out
    inline __device__ void local_fifo_update_finished(vidType u, int warp_id) {
        // about the thread operating on this function
        int thread_lane = threadIdx.x & (WARP_SIZE-1);
        
        // iterate through all the destinations to find u
        for (int local_row_id = 0; local_row_id < ndevices - 1; local_row_id++) {
            int local_col_id;
            if (local_fifo_find_updating_col(&local_col_id, local_row_id, u, warp_id) == 1) { // found
                if (thread_lane == 0) {
                    local_normal_warp_updating[local_row_id * LOCAL_FIFO_EACH_DEST_LEN + local_col_id] = 0;
                    __threadfence();
                    local_ready_for_sender[local_row_id * LOCAL_FIFO_EACH_DEST_LEN + local_col_id] = 1;
                    __threadfence();
                }
            }
        }
    }

    // grab a not occupied buffer_id from the remote GPU
    inline __device__ int remote_fifo_get_empty_space(int dest, int channel, int warp_id) {
        // about the thread operating on this function
        int thread_lane = threadIdx.x & (WARP_SIZE-1);

        // we start from the hash of the warp_id
        for (int find_col = hash(warp_id, GLOBAL_FIFO_EACH_CHANNEL_LEN); ; find_col = (find_col + 1) % GLOBAL_FIFO_EACH_CHANNEL_LEN) {
            int aquire_result;
            if (thread_lane == 0) {
                aquire_result = nvshmem_int_atomic_swap(&global_occupied[channel * GLOBAL_FIFO_EACH_CHANNEL_LEN + find_col], 1, dest);
            }
            aquire_result = __shfl_sync(0xffffffff, aquire_result, 0);
            if (aquire_result == 0) {   // found an empty space
                return find_col;
            }
        }
    }

    // how the senders check the local fifo and send messages to remote GPUs
    inline __device__ void check_local(GraphNVSHMEM g, int warp_id, int n_normal_blocks) {
        // about the thread operating on this function
        int thread_lane = threadIdx.x & (WARP_SIZE-1);
        int warp_lane = threadIdx.x / WARP_SIZE;
#ifdef SENDER_START
        if (thread_lane == 0) {
            nvshmem_fence();
            while (nvshmem_int_atomic_swap(print_lock, 1, 0) == 1)
                ;
            printf("pe %d sender %d start.\n", fifo_id, warp_id);
            nvshmem_int_atomic_swap(print_lock, 0, 0);
        }
#endif
        
        // assign a row which it will take care of
        int local_row_id = warp_id % (ndevices-1);
        int dest = (local_row_id < fifo_id) ? local_row_id : local_row_id+1;

        // assign a channel which it will put the message to
        int channel = (warp_id / (ndevices-1)) % NUM_CHANNELS;

        // poll the message line
        for (int local_col_id = 0; ; local_col_id = (local_col_id + 1) % LOCAL_FIFO_EACH_DEST_LEN) {
            // check whether it is time to end the checking
            __threadfence();
            if (*normal_warps_finished_counter == 0) {
                if (blockIdx.x == n_normal_blocks && warp_lane == 0) {    // it is the first among all the sender blocks
                    for (int finish_signal_dest = 0; finish_signal_dest < ndevices; finish_signal_dest++) {
                        if (finish_signal_dest == fifo_id) continue;
                        int buffer_id = remote_fifo_get_empty_space(finish_signal_dest, channel, warp_id);
                        if (thread_lane == 0) {
                            nvshmem_int_p(&global_message_type_buffer[channel * GLOBAL_FIFO_EACH_CHANNEL_LEN + buffer_id], 1, finish_signal_dest);
                            nvshmem_fence();
                            nvshmem_int_p(&global_copy_finished_signal_buffer[channel * GLOBAL_FIFO_EACH_CHANNEL_LEN + buffer_id], 1, finish_signal_dest);
                            nvshmem_fence();
#ifdef SENDER_FINISH_SIGNAL
                            while (nvshmem_int_atomic_swap(print_lock, 1, 0) == 1)
                                ;
                            printf("sender in %d finished and telling %d\n", fifo_id, finish_signal_dest);
                            nvshmem_int_atomic_swap(print_lock, 0, 0);
#endif
                        }
                    }
                }
                return;
            }
            
            int aquire_result;
            if (thread_lane == 0) {
                aquire_result = atomicExch(&local_ready_for_sender[local_row_id * LOCAL_FIFO_EACH_DEST_LEN + local_col_id], 0);
            }
            aquire_result = __shfl_sync(0xffffffff, aquire_result, 0);
            if (aquire_result == 1) {   // successfully get the ready message
                // send it to the remote gpu
                int buffer_id = remote_fifo_get_empty_space(dest, channel, warp_id);
                vidType u = local_u_buffer[local_row_id * LOCAL_FIFO_EACH_DEST_LEN + local_col_id];
                vidType task_count = local_task_count_buffer[local_row_id * LOCAL_FIFO_EACH_DEST_LEN + local_col_id];
                if (thread_lane == 0) {     // send the "single" message
                    nvshmem_int_p(&global_message_type_buffer[channel * GLOBAL_FIFO_EACH_CHANNEL_LEN + buffer_id], 0, dest);
                    nvshmem_int32_p(&global_u_buffer[channel * GLOBAL_FIFO_EACH_CHANNEL_LEN + buffer_id], u, dest);
                    nvshmem_int32_p(&global_task_count_buffer[channel * GLOBAL_FIFO_EACH_CHANNEL_LEN + buffer_id], task_count, dest);
                    nvshmem_int32_p(&global_degree_buffer[channel * GLOBAL_FIFO_EACH_CHANNEL_LEN + buffer_id], g.get_degree(u), dest);
                }
                nvshmemx_int32_put_warp(&global_tasks_buffer[channel * GLOBAL_FIFO_EACH_CHANNEL_LEN * max_degree + buffer_id * max_degree], &local_tasks_buffer[local_row_id * LOCAL_FIFO_EACH_DEST_LEN * max_degree + local_col_id * max_degree], task_count, dest);
                nvshmemx_int32_put_warp(&global_list_buffer[channel * GLOBAL_FIFO_EACH_CHANNEL_LEN * max_degree + buffer_id * max_degree], g.N(u), g.get_degree(u), dest);
                nvshmem_fence();
                nvshmem_int_p(&global_copy_finished_signal_buffer[channel * GLOBAL_FIFO_EACH_CHANNEL_LEN + buffer_id], 1, dest);
                nvshmem_fence();

                // debug
#ifdef PRINT_SENDER
                if (thread_lane == 0) {
                    nvshmem_fence();
                    while (nvshmem_int_atomic_swap(print_lock, 1, 0) == 1)
                        ;
                    printf("sender in %d got u=%d to send to %d\n", fifo_id, u, dest);
                    for (int task_id = 0; task_id < task_count; task_id++) {
                        printf(" %d", local_tasks_buffer[local_row_id * LOCAL_FIFO_EACH_DEST_LEN * max_degree + local_col_id * max_degree + task_id]);
                    }
                    printf("\n");
                    nvshmem_int_atomic_swap(print_lock, 0, 0);
                }
#endif

                // clear the local fifo buffer
                if (thread_lane == 0) {
                    local_occupied[local_row_id * LOCAL_FIFO_EACH_DEST_LEN + local_col_id] = 0;
                    local_task_count_buffer[local_row_id * LOCAL_FIFO_EACH_DEST_LEN + local_col_id] = 0;
                    __threadfence();
                }
            }
        }
    }

    // recvers receive messages
    inline __device__ int recv(int warp_id, vidType *u, vidType *task_count, vidType **tasks, vidType *degree, vidType **list, int *channel_res, int *buffer_id_res) {
        // about the thread operating on this function
        int thread_lane = threadIdx.x & (WARP_SIZE-1);
        
        // we assign a channel for the warp operating on this function
        int channel = warp_id % NUM_CHANNELS;

        // iterate throught the channel to find the new messages
        for (int buffer_id = 0; ; buffer_id = (buffer_id + 1) % GLOBAL_FIFO_EACH_CHANNEL_LEN) {
            // check the peer finished counter to decide whether other warps have finished sending
            __threadfence();
            if (*peers_finished_counter == 0) {     // it's time to stop
                return 1;   // end signal to outer side
            }

            // try to capture this message
            nvshmem_fence();
            int aquire_result;
            if (thread_lane == 0) {
                aquire_result = atomicExch(&global_copy_finished_signal_buffer[channel * GLOBAL_FIFO_EACH_CHANNEL_LEN + buffer_id], 0);
            }
            aquire_result = __shfl_sync(0xffffffff, aquire_result, 0);
            if (aquire_result == 1) {   // successfully get a new message
                nvshmem_fence();
                if (global_message_type_buffer[channel * GLOBAL_FIFO_EACH_CHANNEL_LEN + buffer_id] == 1) {  // finish message
                    if (thread_lane == 0) {
                        atomicSub(peers_finished_counter, 1);
                    }
                } else {
                    *u = global_u_buffer[channel * GLOBAL_FIFO_EACH_CHANNEL_LEN + buffer_id];
                    *task_count = global_task_count_buffer[channel * GLOBAL_FIFO_EACH_CHANNEL_LEN + buffer_id];
                    *tasks = &global_tasks_buffer[channel * GLOBAL_FIFO_EACH_CHANNEL_LEN * max_degree + buffer_id * max_degree];
                    *degree = global_degree_buffer[channel * GLOBAL_FIFO_EACH_CHANNEL_LEN + buffer_id];
                    *list = &global_list_buffer[channel * GLOBAL_FIFO_EACH_CHANNEL_LEN * max_degree + buffer_id * max_degree];
                    *channel_res = channel;
                    *buffer_id_res = buffer_id;

                    return 0;
                }
            }
        }
    }

    inline __device__ void release_message_buffer(int channel, int buffer_id) {
        global_occupied[channel * GLOBAL_FIFO_EACH_CHANNEL_LEN + buffer_id] = 0;
        nvshmem_fence();
    }
};
