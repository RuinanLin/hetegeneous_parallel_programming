#pragma once
#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <stdlib.h>

#define WARP_SIZE   32

#define CUDA_SAFE_CALL(stmt)                                                          \
    do {                                                                          \
        cudaError_t result = (stmt);                                              \
        if (cudaSuccess != result) {                                              \
            fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, \
                    cudaGetErrorString(result));                                  \
            exit(-1);                                                             \
        }                                                                         \
    } while (0)

typedef int32_t vidType;

class NVSHMEMBufferWarp {
private:
    // parameters
    int my_id;
    int ndevices;
    int slot_size;                  // how many elements this slot can hold

    // recv
    int *valid;                     // [ ndevices-1 ]   // every time a remote GPU's sender finishes writing data, the valid[dest_row_id_here] will be set
    uint64_t *valid_signal_count;   // [ 1 ]            // count the number of valid signals. it is useful when the recver warp checks whether there are new messages

    // send
    uint64_t *ready;                // [ ndevices-1 ]   // every time a remote GPU's recver finishes dealing with the data, the ready[dest_row_id_here] will be set

    // termination
    int *num_working_producers;     // [ 1 ]            // how many producer warps are still going to produce new messages
    
    // content
    int *msg_type;                  // [ ndevices-1 ]   // 0: normal; 1: exit
    int *msg_len;                   // [ ndevices-1 ]
    vidType *content;               // [ ndevices-1, slot_size ]

    // debug
    int *counter;

public:
    NVSHMEMBufferWarp(int mype, int num_devices, int slot_sz, int num_producers) : my_id(mype), ndevices(num_devices), slot_size(slot_sz) {
        valid = (int *)nvshmem_malloc((ndevices-1) * sizeof(int));
        valid_signal_count = (uint64_t *)nvshmem_malloc(sizeof(uint64_t));
        ready = (uint64_t *)nvshmem_malloc((ndevices-1) * sizeof(uint64_t));
        CUDA_SAFE_CALL(cudaMalloc((void **)&num_working_producers, sizeof(int)));
        msg_type = (int *)nvshmem_malloc(sizeof(int));
        msg_len = (int *)nvshmem_malloc((ndevices-1) * sizeof(int));
        content = (vidType *)nvshmem_malloc((ndevices-1) * slot_size * sizeof(vidType));
        CUDA_SAFE_CALL(cudaMalloc((void **)&counter, sizeof(int)));

        int *valid_h = (int *)malloc((ndevices-1) * sizeof(int));
        for (int i = 0; i < ndevices-1; i++) {
            valid_h[i] = 0;
        }
        CUDA_SAFE_CALL(cudaMemcpy(valid, valid_h, (ndevices-1) * sizeof(int), cudaMemcpyHostToDevice));

        uint64_t valid_signal_count_h = 0;
        CUDA_SAFE_CALL(cudaMemcpy(valid_signal_count, &valid_signal_count_h, sizeof(uint64_t), cudaMemcpyHostToDevice));

        uint64_t *ready_h = (uint64_t *)malloc((ndevices-1) * sizeof(uint64_t));
        for (int i = 0; i < ndevices-1; i++) {
            ready_h[i] = 1;
        }
        CUDA_SAFE_CALL(cudaMemcpy(ready, ready_h, (ndevices-1) * sizeof(uint64_t), cudaMemcpyHostToDevice));

        CUDA_SAFE_CALL(cudaMemcpy(num_working_producers, &num_producers, sizeof(int), cudaMemcpyHostToDevice));

        int counter_h = 0;
        CUDA_SAFE_CALL(cudaMemcpy(counter, &counter_h, sizeof(int), cudaMemcpyHostToDevice));
    }

    inline __device__ int get_my_id() { return my_id; }
    inline __device__ int get_ndevices() { return ndevices; }

    __device__ void producer_write_msg(int message_size, vidType *message, int dest_id, int type) {
        int thread_lane = threadIdx.x & (WARP_SIZE-1);
        
        int dest_row_id_here = (dest_id < my_id)? dest_id : dest_id - 1;
        nvshmem_signal_wait_until(&ready[dest_row_id_here], NVSHMEM_CMP_EQ, 1);
        ready[dest_row_id_here] = 0;
        nvshmem_fence();    // can it be deleted?
        
        int my_row_id_there = (my_id < dest_id)? my_id : my_id - 1;
        if (thread_lane == 0) {
            nvshmem_int_p(&msg_type[my_row_id_there], type, dest_id);
        }
        __syncwarp();   // can it be deleted?
        if (type == 0) {
            if (thread_lane == 0) {
                nvshmem_int_p(&msg_len[my_row_id_there], message_size, dest_id);
            }
            __syncwarp();
            nvshmemx_int32_put_warp(&content[my_row_id_there * slot_size], message, message_size, dest_id);
            nvshmem_fence();    // can it be deleted?
        }
        
        if (thread_lane == 0) {
            nvshmem_int_p(&valid[my_row_id_there], 1, dest_id);
            nvshmemx_signal_op(valid_signal_count, 1, NVSHMEM_SIGNAL_ADD, dest_id);
            nvshmem_fence();    // can it be deleted?
        }
        __syncwarp();
    }

    __device__ int consumer_get_msg_pointer(vidType **start, int *degree, int *dest) {
        int thread_lane = threadIdx.x & (WARP_SIZE-1);              // thread index within the warp

        while (1) {
            // first, check whether all the sender warps have exited
            if (*num_working_producers == 0) {
                return 1;
            }

            // wait for a new message
            nvshmem_signal_wait_until(valid_signal_count, NVSHMEM_CMP_GT, 0);

            // find the new message
            for (int dest_id = 0; dest_id < ndevices; dest_id++) {
                if (dest_id == my_id) continue;
                int dest_row_id_here = (dest_id < my_id)? dest_id : dest_id-1;
                nvshmem_fence();    // can it be deleted?
                if (valid[dest_row_id_here] == 0) continue;
                if (thread_lane == 0) {
                    valid[dest_row_id_here] = 0;
                    nvshmem_uint64_atomic_add(valid_signal_count, -1, my_id);  // can we change it back to normal?
                    nvshmem_fence();    // can it be deleted?
                }
                __syncwarp();   // can it be deleted?

                nvshmem_fence();    // can it be deleted?

                if (msg_type[dest_row_id_here] == 1) {
                    if (thread_lane == 0) {
                        (*num_working_producers)--;
                    }
                    __syncwarp();   // can it be deleted?
                    consumer_release(dest_id);
                } else {
                    *degree = msg_len[dest_row_id_here];
                    *start = &content[dest_row_id_here * slot_size];
                    *dest = dest_id;
                    return 0;
                }
            }
        }
    }

    __device__ void consumer_release(int dest_id) {
        int my_row_id_there = (my_id < dest_id)? my_id : my_id - 1;
        int thread_lane = threadIdx.x & (WARP_SIZE-1);
        if (thread_lane == 0) {
            nvshmemx_signal_op(&ready[my_row_id_there], 1, NVSHMEM_SIGNAL_SET, dest_id);
        }
        __syncwarp();
    }
};