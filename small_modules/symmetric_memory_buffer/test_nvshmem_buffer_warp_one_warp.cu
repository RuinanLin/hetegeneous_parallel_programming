#include "nvshmem_buffer_warp.h"
#include "nvshmem.h"
#include "nvshmemx.h"
#include <iostream>

#define BLOCK_SIZE  256

#define SLOT_SIZE   16

#define SEND_ROUND  8

__device__ void sender_launch(NVSHMEMBufferWarp nvshmem_buffer_warp, vidType *message_scratchpad) {
    int thread_lane = threadIdx.x & (WARP_SIZE-1);              // thread index within the warp

    int count = 0;
    
    for (int round = 0; round < SEND_ROUND; round++) {
        for (int dest = 0; dest < nvshmem_buffer_warp.get_ndevices(); dest++) {
            if (dest == nvshmem_buffer_warp.get_my_id()) continue;
            int degree = ((round + dest) * 137) % SLOT_SIZE;
            if (thread_lane == 0) {
                for (int i = 0; i < degree; i++) {
                    message_scratchpad[i] = degree - i;
                    count += degree - i;
                }
            }
            __syncwarp();
            nvshmem_buffer_warp.producer_write_msg(degree, message_scratchpad, dest, 0);
        }
    }
    if (thread_lane == 0) {
        printf("pe %d sender count = %d\n", nvshmem_buffer_warp.get_my_id(), count);
    }
    __syncwarp();
    for (int dest = 0; dest < nvshmem_buffer_warp.get_ndevices(); dest++) {
        if (dest == nvshmem_buffer_warp.get_my_id()) continue;
        nvshmem_buffer_warp.producer_write_msg(0, 0, dest, 1);
    }
}

__device__ void recver_launch(NVSHMEMBufferWarp nvshmem_buffer_warp) {
    int thread_lane = threadIdx.x & (WARP_SIZE-1);

    int count = 0;
    
    vidType *start;
    int degree;
    int dest;
    while (1) {
        // if (thread_lane == 0) {
        //     printf("hello.\n");
        // }
        // __syncwarp();
        if (nvshmem_buffer_warp.consumer_get_msg_pointer(&start, &degree, &dest) == 1) {
            if (thread_lane == 0) {
                printf("pe %d recver count = %d\n", nvshmem_buffer_warp.get_my_id(), count);
            }
            __syncwarp();
            return;
        }
        for (int i = 0; i < degree; i++) {
            if (start[i] != degree - i) {
                if (thread_lane == 0) {
                    printf("pe %d start[%d] = %d, should be %d.\n", nvshmem_buffer_warp.get_my_id(), i, start[i], degree - i);
                }
                __syncwarp();
            }
            if (thread_lane == 0) {
                count += start[i];
            };
            __syncwarp();
        }
        nvshmem_buffer_warp.consumer_release(dest);
    }
}

__global__ void test_nvshmem_buffer_warp(NVSHMEMBufferWarp nvshmem_buffer_warp) {
    int warp_lane = threadIdx.x / WARP_SIZE;                    // warp index within the CTA

    __shared__ vidType message_scratchpad[SLOT_SIZE];

    if (warp_lane == 0) {
        sender_launch(nvshmem_buffer_warp, message_scratchpad);
    } else {
        recver_launch(nvshmem_buffer_warp);
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

    // initialize NVSHMEMBufferWarp
    int mype = nvshmem_my_pe();
    NVSHMEMBufferWarp nvshmem_buffer_warp(mype, ndevices, SLOT_SIZE, ndevices-1);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    nvshmem_barrier_all();

    // launch kernel
    test_nvshmem_buffer_warp<<<1, WARP_SIZE*2>>>(nvshmem_buffer_warp);  // warp 0 is a sender, warp 1 is a recver
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    return 0;
}