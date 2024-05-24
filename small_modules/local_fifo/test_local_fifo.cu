#include "local_fifo.h"

#define NUM_SLOTS               800
#define SLOT_SIZE               100
#define BLOCK_SIZE              256
#define NUM_WARPS_IN_A_BLOCK    8

#define NUM_PRODUCER_BLOCKS     10
#define NUM_CONSUMER_BLOCKS     5

#define PRODUCER_SEND_ROUNDS    8

__device__ void producer(LocalFIFO fifo) {
    int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;    // global thread index
    int warp_id = thread_id / WARP_SIZE;                        // global warp index
    int thread_lane = threadIdx.x & (WARP_SIZE-1);              // thread index within the warp

    if (thread_lane == 0) {
        printf("producer %d starts.\n", warp_id);
    }
    __syncwarp();

    int degree = warp_id % SLOT_SIZE;
    for (int round = 0; round < PRODUCER_SEND_ROUNDS; round++) {
        int slot_id = fifo.producer_get_empty_slot_id_warp();
        // if (thread_lane == 0) {
        //     printf("producer %d got slot_id %d.\n", warp_id, slot_id);
        // }
        __syncwarp();
        for (int i = 0; i < degree; i++) {
            fifo.producer_update_msg_content_warp(slot_id, i, degree - i);
        }
        fifo.producer_update_msg_size_warp(slot_id, degree);
        fifo.producer_update_msg_finish_warp(slot_id);
        // if (thread_lane == 0) {
        //     printf("producer %d released slot_id %d.\n", warp_id, slot_id);
        // }
        __syncwarp();
    }
    fifo.producer_exit();

    if (thread_lane == 0) {
        printf("producer %d exit.\n", warp_id);
    }
    __syncwarp();
}

__device__ void consumer(LocalFIFO fifo) {
    int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;    // global thread index
    int warp_id = thread_id / WARP_SIZE;                        // global warp index
    int thread_lane = threadIdx.x & (WARP_SIZE-1);              // thread index within the warp

    if (thread_lane == 0) {
        printf("consumer %d starts.\n", warp_id);
    }
    __syncwarp();

    while (1) {
        int slot_id = fifo.consumer_get_full_slot_id_warp();
        if (slot_id == -1) {    // it's time to finish
            if (thread_lane == 0) {
                printf("consumer %d exit.\n", warp_id);
            }
            __syncwarp();
            return;
        }
        int degree = fifo.consumer_read_msg_size_warp(slot_id);
        vidType *msg_pointer = fifo.consumer_get_msg_pointer_warp(slot_id);
        for (int i = 0; i < degree; i++) {
            if (msg_pointer[i] != degree - i) {
                if (thread_lane == 5) {
                    printf("error: msg[%d][%d] = %d, should be %d, degree = %d, %d - %d = %d\n", slot_id, i, (int)(msg_pointer[i]), degree - i, degree, degree, i, degree - i);
                    // printf("%d - %d = %d\n", degree, i, degree - i);
                }
                __syncwarp();
            }
            // assert(msg_pointer[i] == degree - i);
        }
        fifo.consumer_read_msg_finish_warp(slot_id);
        __syncwarp();
    }
}

__global__ void test_local_fifo(LocalFIFO fifo) {
    // go to different tasks
    if (blockIdx.x < NUM_PRODUCER_BLOCKS) {
        producer(fifo);
    } else {
        consumer(fifo);
    }
}

int main() {
    LocalFIFO fifo(NUM_SLOTS, SLOT_SIZE, NUM_PRODUCER_BLOCKS * NUM_WARPS_IN_A_BLOCK);
    test_local_fifo<<<NUM_PRODUCER_BLOCKS+NUM_CONSUMER_BLOCKS, BLOCK_SIZE>>>(fifo);
    cudaDeviceSynchronize();
    return 0;
}