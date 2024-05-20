#include <stdio.h>
#include <stdlib.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <iostream>

#define CUDA_SAFE_CALL(stmt)                                                          \
    do {                                                                          \
        cudaError_t result = (stmt);                                              \
        if (cudaSuccess != result) {                                              \
            fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, \
                    cudaGetErrorString(result));                                  \
            exit(-1);                                                             \
        }                                                                         \
    } while (0)

#define WARP_SIZE   32

__global__ void read_and_stop(uint64_t *end, uint64_t *signal) {
    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();

    if (mype == 0) {
        for (int pe_id = 1; pe_id < npes; pe_id++) {
            nvshmem_uint64_p(end, 1, pe_id);
            nvshmemx_signal_op(signal, 1, NVSHMEM_SIGNAL_SET, pe_id);
            printf("%d sent.\n", pe_id);
        }
    } else {
        nvshmem_signal_wait_until(signal, NVSHMEM_CMP_NE, 0);
        // int my_end;
        // do {
        //     nvshmem_fence();
        //     my_end = *end;
        // } while (my_end == 0);
        printf("%d stopped, my_end = %ld.\n", mype, *end);
    }
}

int main(int c, char *v[]) {
    int mype_node;

    nvshmem_init();
    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    CUDA_SAFE_CALL(cudaSetDevice(mype_node));

    // allocate memory
    uint64_t h_end = 0;
    uint64_t *d_end = (uint64_t *)nvshmem_malloc(sizeof(uint64_t));
    CUDA_SAFE_CALL(cudaMemcpy(d_end, &h_end, sizeof(uint64_t), cudaMemcpyHostToDevice));
    uint64_t h_signal = 0;
    uint64_t *d_signal = (uint64_t *)nvshmem_malloc(sizeof(uint64_t));
    CUDA_SAFE_CALL(cudaMemcpy(d_signal, &h_signal, sizeof(uint64_t), cudaMemcpyHostToDevice));

    // launch the kernel
    dim3 gridDim(1), blockDim(1);
    void *args[] = {&d_end, &d_signal};
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    nvshmem_barrier_all();
    nvshmemx_collective_launch((const void *)read_and_stop, gridDim, blockDim, args, 0, 0);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    nvshmem_barrier_all();
    return 0;
}