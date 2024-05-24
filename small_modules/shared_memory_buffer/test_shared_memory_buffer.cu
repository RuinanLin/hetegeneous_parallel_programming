#include <stdio.h>
#include <stdint.h>
#include <assert.h>

#define BLOCK_SIZE              256
#define WARP_SIZE               32

#define PRODUCER_SEND_ROUNDS    8
#define SLOT_SIZE               128
#define NUM_WARPS_IN_A_BLOCK    8

typedef int32_t vidType;

__device__ void worker_launch(int *valid_for_sender, int *num_working_workers, int *msg_len, vidType *msg_buffer, vidType *sum_test) {
    // about the threads and blocks
    int thread_lane = threadIdx.x & (WARP_SIZE-1);
    int warp_lane = threadIdx.x / WARP_SIZE;            // warp index within the CTA

    for (int round = 0; round < PRODUCER_SEND_ROUNDS; round++) {
        int degree = (warp_lane * 37 + round) % SLOT_SIZE;
        while (1) {
            __threadfence_block();
            if (*valid_for_sender == 0) {
                break;
            }
        }
        if (thread_lane == 0) {
            for (int i = 0; i < degree; i++) {
                msg_buffer[i] = degree - i;
                atomicAdd(sum_test, degree - i);
            }
            *msg_len = degree;
            __threadfence_block();
            *valid_for_sender = 1;
            __threadfence_block();
        }
        __syncwarp();
    }
    if (thread_lane == 0) {
        atomicSub(num_working_workers, 1);
        __threadfence_block();
    }

    if (thread_lane == 0) {
        printf("warp %d exits.\n", warp_lane);
    }
}

__device__ void sender_launch(int *valid_for_sender, int *num_working_workers, int *msg_len, vidType *msg_buffer, vidType *sum_test) {
    int thread_lane = threadIdx.x & (WARP_SIZE-1);
    
    while (1) {
        int done = 0;
        __threadfence_block();
        if (*num_working_workers == 0) {
            done = 1;
        }

        for (int worker_id = 0; worker_id < NUM_WARPS_IN_A_BLOCK - 1; worker_id++) {
            __threadfence_block();
            if (valid_for_sender[worker_id] == 1) {
                int degree = msg_len[worker_id];
                for (int i = 0; i < degree; i++) {
                    if (msg_buffer[worker_id * SLOT_SIZE + i] != degree - i && thread_lane == 5) {
                        printf("msg_buffer[%d][%d] = %d, should be %d.\n", worker_id, i, (int)msg_buffer[i], degree - i);
                    }
                    if (thread_lane == 0) {
                        atomicSub(sum_test, degree - i);
                    }
                    __syncwarp();
                }

                __syncwarp();
                __threadfence_block();

                if (thread_lane == 0) {
                    valid_for_sender[worker_id] = 0;
                    __threadfence_block();
                }
                __syncwarp();
            }
        }

        if (done == 1) {
            break;
        }
    }

    if (thread_lane == 0) {
        __threadfence_block();
        printf("sum_test = %d\n", (int)*sum_test);

        printf("warp 7 exits.\n");
    }
}

__global__ void test_shared_memory_buffer(int slot_size) {
    // about the threads and blocks
    int thread_lane = threadIdx.x & (WARP_SIZE-1);
    int warp_lane = threadIdx.x / WARP_SIZE;            // warp index within the CTA
    
    __shared__ int valid_for_sender[NUM_WARPS_IN_A_BLOCK - 1];
    __shared__ int num_working_workers;
    __shared__ int msg_len[NUM_WARPS_IN_A_BLOCK - 1];
    __shared__ vidType msg_buffer[NUM_WARPS_IN_A_BLOCK * SLOT_SIZE];
    __shared__ vidType sum_test;

    __syncthreads();

    if (warp_lane == 0 && thread_lane == 0) {
        num_working_workers = NUM_WARPS_IN_A_BLOCK - 1;
        sum_test = 0;
    }

    if (warp_lane < NUM_WARPS_IN_A_BLOCK - 1 && thread_lane == 0) {
        valid_for_sender[warp_lane] = 0;
    }
    __threadfence_block();

    __syncthreads();

    if (warp_lane < NUM_WARPS_IN_A_BLOCK - 1) {
        worker_launch(valid_for_sender + warp_lane, &num_working_workers, msg_len + warp_lane, msg_buffer + warp_lane * SLOT_SIZE, &sum_test);
    } else {
        sender_launch(valid_for_sender, &num_working_workers, msg_len, msg_buffer, &sum_test);
    }
}

int main() {
    test_shared_memory_buffer<<<1, BLOCK_SIZE>>>(SLOT_SIZE);
    cudaDeviceSynchronize();
    return 0;
}