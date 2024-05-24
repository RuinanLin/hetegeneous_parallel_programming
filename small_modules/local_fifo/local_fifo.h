#pragma once
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>

/*********************************************************************/

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

typedef int64_t vidType;

/*********************************************************************/

class LocalFIFO {
private:
    // parameters
    int n_slots;
    int slot_size;

    // controllers
    volatile int *lock;
    volatile unsigned long long int *head;
    volatile unsigned long long int *tail;
    int *ready_for_consumer;    // [ n_slots ]
    int *n_working_producers;

    // content
    int *msg_size;      // [ n_slots ]
    vidType *msg_buffer;           // [ n_slots, slot_size ]

public:
    LocalFIFO(int num_slots, int slot_sz, int num_producers) : n_slots(num_slots), slot_size(slot_sz), lock(NULL), head(NULL), tail(NULL), msg_size(NULL), msg_buffer(NULL) {
        CUDA_SAFE_CALL(cudaMalloc((void **)&lock, sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void **)&head, sizeof(unsigned long long int)));
        CUDA_SAFE_CALL(cudaMalloc((void **)&tail, sizeof(unsigned long long int)));
        CUDA_SAFE_CALL(cudaMalloc((void **)&ready_for_consumer, n_slots * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void **)&n_working_producers, sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void **)&msg_size, n_slots * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void **)&msg_buffer, n_slots * slot_size * sizeof(vidType)));

        int lock_h = 0;
        CUDA_SAFE_CALL(cudaMemcpy((int *)lock, &lock_h, sizeof(int), cudaMemcpyHostToDevice));
        unsigned long long int head_h = 0;
        CUDA_SAFE_CALL(cudaMemcpy((unsigned long long int *)head, &head_h, sizeof(unsigned long long int), cudaMemcpyHostToDevice));
        unsigned long long int tail_h = 0;
        CUDA_SAFE_CALL(cudaMemcpy((unsigned long long int *)tail, &tail_h, sizeof(unsigned long long int), cudaMemcpyHostToDevice));
        int *ready_for_consumer_h = (int *)malloc(n_slots * sizeof(int));
        for (int i = 0; i < n_slots; i++) ready_for_consumer_h[i] = 0;
        CUDA_SAFE_CALL(cudaMemcpy(ready_for_consumer, ready_for_consumer_h, n_slots * sizeof(int), cudaMemcpyHostToDevice));
        int n_working_producers_h = num_producers;
        CUDA_SAFE_CALL(cudaMemcpy(n_working_producers, &n_working_producers_h, sizeof(int), cudaMemcpyHostToDevice));
    }

    inline __device__ int my_thread_lane() {
        return (threadIdx.x & (WARP_SIZE-1));
    }

    inline __device__ void acquire_lock_warp() {
        if (my_thread_lane() == 0) {
            while (atomicExch((int *)lock, 1))
                ;
        }
        __syncwarp();
    }

    inline __device__ void release_lock_warp() {
        if (my_thread_lane() == 0) {
            atomicExch((int *)lock, 0);
        }
        __syncwarp();
    }

    // must acquire lock before calling this function
    __device__ int is_full_warp() {
        __threadfence();
        int result = (*head + n_slots == *tail) ? 1 : 0;
        return result;
    }

    __device__ int producer_get_empty_slot_id_warp() {
        while (1) {
            acquire_lock_warp();
            
            if (is_full_warp()) {   // unfortunately cannot get an empty slot_id
                release_lock_warp();
                continue;
            }

            __threadfence();
            int slot_id = (*tail) % n_slots;
            if (my_thread_lane() == 0) {    // update tail
                atomicAdd((unsigned long long int *)tail, 1);
                __threadfence();
                printf("tail = %ld.\n", *tail);
            }
            __syncwarp();

            __threadfence();
            release_lock_warp();
            return slot_id;
        }
    }

    // this function does not require a lock
    __device__ void producer_update_msg_content_warp(int slot_id, int content_id, vidType value) {
        __threadfence();
        if (my_thread_lane() == 0) {
            msg_buffer[slot_id * slot_size + content_id] = value;
        }
        __syncwarp();
        __threadfence();
    }

    // this function does not require a lock
    __device__ void producer_update_msg_size_warp(int slot_id, int size) {
        __threadfence();
        if (my_thread_lane() == 0) {
            msg_size[slot_id] = size;
        }
        __syncwarp();
        __threadfence();
    }

    // this function does not require a lock
    __device__ void producer_update_msg_finish_warp(int slot_id) {
        __threadfence();
        if (my_thread_lane() == 0) {
            ready_for_consumer[slot_id] = 1;
        }
        __syncwarp();
        __threadfence();
    }

    __device__ void producer_exit() {
        __threadfence();
        if (my_thread_lane() == 0) {
            atomicSub(n_working_producers, 1);
        }
        __syncwarp();
        __threadfence();
    }

    __device__ int all_producers_have_finished() {
        __threadfence();
        if (*n_working_producers == 0) {
            __threadfence();
            return 1;
        }
        __threadfence();
        return 0;
    }

    __device__ int consumer_get_full_slot_id_warp() {
        while (1) {
            if (all_producers_have_finished() == 1) {
                return -1;
            }

            acquire_lock_warp();

            __threadfence();
            if (*head == *tail || ready_for_consumer[(*head) % n_slots] == 0) { // unfortunately there are no new messages to read
                release_lock_warp();
                continue;
            }

            int slot_id = (*head) % n_slots;
            if (my_thread_lane() == 0) {
                atomicAdd((unsigned long long int *)head, 1);
                __threadfence();
                printf("head = %ld.\n", *head);
            }
            __syncwarp();

            __threadfence();
            release_lock_warp();
            return slot_id;
        }
    }

    // this function does not require a lock
    __device__ int consumer_read_msg_size_warp(int slot_id) {
        __threadfence();
        return msg_size[slot_id];
    }

    // this function does not require a lock
    __device__ vidType *consumer_get_msg_pointer_warp(int slot_id) {
        return &msg_buffer[slot_id * slot_size];
    }

    __device__ void consumer_read_msg_finish_warp(int slot_id) {
        __threadfence();
        if (my_thread_lane() == 0) {
            ready_for_consumer[slot_id] = 0;
        }
        __syncwarp();
        __threadfence();
    }
};