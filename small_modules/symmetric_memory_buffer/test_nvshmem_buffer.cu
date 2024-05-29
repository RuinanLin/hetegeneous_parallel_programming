#include "nvshmem_buffer.h"
#include "nvshmem.h"
#include "nvshmemx.h"
#include <iostream>

#define BLOCK_SIZE  256

#define SLOT_SIZE   16

#define SEND_ROUND  8

#define NUM_WARP_GROUPS 8

__device__ void sender_launch(NVSHMEMBuffer nvshmem_buffer, vidType *message_scratchpad) {
    int thread_lane = threadIdx.x & (WARP_SIZE-1);  // thread index within the warp
    int warp_lane = threadIdx.x / WARP_SIZE;        // warp index within the CTA
    if (warp_lane != 7) return;

    int count = 0;

    for (int round = 0; round < SEND_ROUND; round++) {
        for (int dest = 0; dest < nvshmem_buffer.get_ndevices(); dest++) {
            if (dest == nvshmem_buffer.get_my_id()) continue;
            int degree = ((round + dest + blockIdx.x) * 137) % SLOT_SIZE;
            if (thread_lane == 0) {
                for (int i = 0; i < degree; i++) {
                    message_scratchpad[i] = degree - i;
                    count += degree - i;
                }
            }
            __syncwarp();
            nvshmem_buffer.producer_write_msg(blockIdx.x, degree, message_scratchpad, dest, 0);
        }
    }
    if (thread_lane == 0) {
        printf("pe %d block %d sender count = %d\n", nvshmem_buffer.get_my_id(), blockIdx.x, count);
    }
    __syncwarp();
    for (int dest = 0; dest < nvshmem_buffer.get_ndevices(); dest++) {
        if (dest == nvshmem_buffer.get_my_id()) continue;
        nvshmem_buffer.producer_write_msg(blockIdx.x, 0, 0, dest, 1);
    }
}

__device__ void recver_launch(NVSHMEMBuffer nvshmem_buffer) {
    int thread_lane = threadIdx.x & (WARP_SIZE-1);  // thread index within the warp
    int warp_lane = threadIdx.x / WARP_SIZE;        // warp index within the CTA

    int count = 0;

    vidType *start;
    int degree;
    int dest;
    while (1) {
        if (nvshmem_buffer.consumer_get_msg_pointer(warp_lane, &start, &degree, &dest) == 1) {
            if (thread_lane == 0) {
                printf("pe %d warp %d recver count = %d\n", nvshmem_buffer.get_my_id(), warp_lane, count);
            }
            __syncwarp();
            return;
        }
        for (int i = 0; i < degree; i++) {
            if (start[i] != degree - i) {
                if (thread_lane == 0) {
                    printf("pe %d warp %d recver start[%d] = %d, should be %d.\n", nvshmem_buffer.get_my_id(), warp_lane, i, start[i], degree - i);
                }
                __syncwarp();
            }
            if (thread_lane == 0) {
                count += start[i];
            }
            __syncwarp();
        }
        nvshmem_buffer.consumer_release(warp_lane, dest);
    }
}

__global__ void test_nvshmem_buffer(NVSHMEMBuffer nvshmem_buffer) {
    __shared__ vidType message_scratchpad[NUM_WARP_GROUPS * SLOT_SIZE];

    if (blockIdx.x < 8) {
        sender_launch(nvshmem_buffer, &message_scratchpad[blockIdx.x * SLOT_SIZE]);
    } else {
        recver_launch(nvshmem_buffer);
    }
}

int main() {
    // initialize nvshmem
    nvshmem_init();
    int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    std::cout << "mype_node = " << mype_node << "\n";
    CUDA_SAFE_CALL(cudaSetDevice(mype_node));

    // get device information
    int ndevices;
    CUDA_SAFE_CALL(cudaGetDeviceCount(&ndevices));

    // initialize NVSHMEMBuffer
    int mype = nvshmem_my_pe();
    NVSHMEMBuffer nvshmem_buffer(mype, ndevices, SLOT_SIZE, ndevices-1, NUM_WARP_GROUPS);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    nvshmem_barrier_all();

    // launch kernel
    test_nvshmem_buffer<<<9, 256>>>(nvshmem_buffer);    // in block 0~7, the last warp is a sender; in block 8, all the warps are recvers
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    return 0;
}