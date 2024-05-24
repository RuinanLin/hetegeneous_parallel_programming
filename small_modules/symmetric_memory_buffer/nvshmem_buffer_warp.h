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
    int slot_size;      // how many elements this slot can hold

    // controller
    uint64_t *valid;    // [ ndevices-1 ]
    uint64_t *ready;    // [ ndevices-1 ]
    
    // content
    vidType *content;   // [ ndevices-1, slot_size ]

public:
    NVSHMEMBufferWarp(int mype, int num_devices, int slot_sz) : my_id(mype), ndevices(num_devices), slot_size(slot_sz) {
        valid = (uint64_t *)nvshmem_malloc((ndevices-1) * sizeof(uint64_t));
        ready = (uint64_t *)nvshmem_malloc((ndevices-1) * sizeof(uint64_t));
        content = (vidType *)nvshmem_malloc((ndevices-1) * slot_size * sizeof(vidType));

        uint64_t *valid_h = (uint64_t *)malloc((ndevices-1) * sizeof(uint64_t));
        for (int i = 0; i < ndevices-1; i++) {
            valid_h[i] = 0;
        }
        CUDA_SAFE_CALL(cudaMemcpy(valid, valid_h, (ndevices-1) * sizeof(uint64_t), cudaMemcpyHostToDevice));

        uint64_t *ready_h = (uint64_t *)malloc((ndevices-1) * sizeof(uint64_t));
        for (int i = 0; i < ndevices-1; i++) {
            ready_h[i] = 1;
        }
        CUDA_SAFE_CALL(cudaMemcpy(ready, ready_h, (ndevices-1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
    }

    __device__ void producer_write_msg(int message_size, vidType *message, int dest_id) {
        int dest_row_id_here = (dest_id < my_id)? dest_id : dest_id - 1;
        nvshmem_signal_wait_until(&ready[dest_row_id_here], NVSHMEM_CMP_EQ, 1);
        ready[dest_row_id_here] = 0;
        nvshmem_fence();
        
        int my_row_id_there = (my_id < dest_id)? my_id : my_id - 1;
        nvshmemx_int32_put_signal_warp(&content[my_row_id_there * slot_size], message, message_size, &valid[my_row_id_there], 1, NVSHMEM_SIGNAL_SET, dest_id);
    }

    __device__ void consumer_get_msg_pointer(vidType **start, vidType **end, int dest_id) {
        int dest_row_id_here = (dest_id < my_id)? dest_id : dest_id - 1;
        nvshmem_signal_wait_until(&valid[dest_row_id_here], NVSHMEM_CMP_EQ, 1);
        valid[dest_row_id_here] = 0;
        nvshmem_fence();

        *start = &content[dest_row_id_here * slot_size];
        *end = &content[dest_row_id_here * slot_size + content[dest_row_id_here * slot_size]];  // we must assume that the first element is the length of the message
    }

    __device__ void consumer_release(int dest_id) {
        int my_row_id_there = (my_id < dest_id)? my_id : my_id - 1;
        int thread_lane = threadIdx.x & (WARP_SIZE-1);
        if (thread_lane == 0) {
            nvshmem_signal_op(&ready[my_row_id_there], 1, NVSHMEM_SIGNAL_SET, dest_id);
        }
        __syncwarp();
    }
};