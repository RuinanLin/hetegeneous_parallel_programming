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

__global__ void read_and_stop(int *end) {
    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();

    if (mype == 0) {
        for (int pe_id = 1; pe_id < npes; pe_id++) {
            nvshmem_int_p(end, 1, pe_id);
            nvshmem_quiet();
            printf("%d sent.\n", pe_id);
        }
    } else {
        int my_end;
        do {
            // my_end = *end;
            my_end = nvshmem_int_g(end, mype);
        } while (my_end != 1);
        printf("%d stopped.\n", mype);
    }
}

int main(int c, char *v[]) {
    int mype_node;

    nvshmem_init();
    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    CUDA_SAFE_CALL(cudaSetDevice(mype_node));

    // allocate memory
    int h_end = 0;
    int *d_end = (int *)nvshmem_malloc(sizeof(int));
    CUDA_SAFE_CALL(cudaMemcpy(d_end, &h_end, sizeof(int), cudaMemcpyHostToDevice));

    // launch the kernel
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    nvshmem_barrier_all();
    read_and_stop<<<1, 1>>>(d_end);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    nvshmem_barrier_all();
    return 0;
}