#include <stdio.h>
#include <assert.h>
#include "shared_memory_buffer.h"

#define BLOCK_SIZE              256

#define PRODUCER_SEND_ROUNDS    8

__device__ void worker_launch(shared_memory_buffer_t *smbuffer) {
    // about the threads and blocks
    int thread_lane = threadIdx.x & (WARP_SIZE-1);
    int warp_lane = threadIdx.x / WARP_SIZE;

    int mype = 2;   // this is an example. assume that all of these happen on device 2
    int npes = 4;   // this is also an example.

    for (int round = 0; round < PRODUCER_SEND_ROUNDS; round++) {
        int degree = (warp_lane * 1379 + round) % PRE_ASSUMED_MAX_DEGREE;
        producer_wait_warp(smbuffer, warp_lane, mype, npes);
        for (int i = 0; i < degree; i++) {
            for (int dest_id = 0; dest_id < npes; dest_id++) {
                if (dest_id == mype) continue;
                producer_write_warp(smbuffer, dest_id, warp_lane, i, degree + dest_id - i, mype);
            }
        }
        for (int dest_id = 0; dest_id < npes; dest_id++) {
            if (dest_id == mype) continue;
            producer_issue_warp(smbuffer, mype, dest_id, warp_lane);
        }
    }
    producer_exit_warp(smbuffer);
    if (thread_lane == 0) {
        printf("warp %d exits.\n", warp_lane);
    }
}

__device__ void sender_launch(shared_memory_buffer_t *smbuffer) {
    int thread_lane = threadIdx.x & (WARP_SIZE-1);

    int sum = 0;

    while (1) {
        vidType *msg_head;
        int msg_length;
        int dest_id;
        int from_warp;
        int mype = 2;
        int npes = 4;
        if (consumer_get_msg_warp(smbuffer, &msg_head, &msg_length, &dest_id, &from_warp, mype, npes) == 1) { // it's time to exit
            break;
        }
        for (int i = 0; i < msg_length; i++) {
            if (msg_head[i] != msg_length + dest_id - i) {
                if (thread_lane == 5) {
                    printf("msg_buffer[%d][%d][%d] = %d, should be %d.\n", dest_id, from_warp, i, msg_head[i], msg_length + dest_id - i);
                }
                __syncwarp();
            }
            sum += msg_head[i];
        }
        consumer_release_msg_warp(smbuffer, dest_id, from_warp, mype);
    }

    if (thread_lane == 0) {
        printf("warp 7 exits.\n");
        printf("sum = %d\n", sum);
    }
}

__global__ void test_shared_memory_buffer() {
    // about the threads and blocks
    int warp_lane = threadIdx.x / WARP_SIZE;

    __shared__ shared_memory_buffer_t smbuffer;

    __syncthreads();    // TODO: can we remove it?

    init_smbuffer_block(&smbuffer);
    if (warp_lane < NUM_WARPS_IN_A_BLOCK - 1) {
        worker_launch(&smbuffer);
    } else {
        sender_launch(&smbuffer);
    }
}

int main() {
    test_shared_memory_buffer<<<1, BLOCK_SIZE>>>();
    cudaDeviceSynchronize();
    return 0;
}