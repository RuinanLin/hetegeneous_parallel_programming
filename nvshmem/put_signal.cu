#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#define WARP_SIZE   32

__global__ void put_signal(int *data, uint64_t *psync, size_t data_len) {
    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();

    // about the threads and blocks
    int thread_lane = threadIdx.x & (WARP_SIZE-1);  // thread index within the warp
    int warp_lane = threadIdx.x / WARP_SIZE;        // warp index within the CTA

    if (thread_lane == 0) {
        if (warp_lane == 0) {
            for (size_t i = 0; i < data_len; i++) {
                data[mype * data_len + i] = mype + i;
            }
            nvshmem_fence();

            nvshmem_signal_wait_until(psync + mype, NVSHMEM_CMP_EQ, npes - 1);
            for (int dest = 0; dest < npes; dest++) {
                if (dest == mype) continue;
                nvshmem_int_put_signal(data + mype * data_len, data + mype * data_len, data_len, psync + mype, 1, NVSHMEM_SIGNAL_SET, dest);
            }
        } else {
            for (int src = 0; src < npes; src++) {
                if (src == mype) continue;

                int sum = 0;
                for (size_t i = 0; i < data_len; i++) {
                    sum += data[src * data_len + i];
                }
                printf("%d, before, from %d : %d\n", mype, src, sum);
                nvshmemx_signal_op(psync + src, 1, NVSHMEM_SIGNAL_ADD, src);
            }

            for (int src = 0; src < npes; src++) {
                if (src == mype) continue;

                nvshmem_signal_wait_until(psync + src, NVSHMEM_CMP_EQ, 1);
                int sum = 0;
                for (size_t i = 0; i < data_len; i++) {
                    sum += data[src * data_len + i];
                }
                printf("%d, after, from %d : %d\n", mype, src, sum);
            }
        }
    }
}

int main(void) {
    size_t data_len = 4;

    nvshmem_init();

    int npes = nvshmem_n_pes();
    int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    
    cudaSetDevice(mype_node);

    int *data = (int *)nvshmem_malloc(sizeof(int) * data_len * npes);
    uint64_t *psync_h = (uint64_t *)malloc(sizeof(uint64_t) * npes);
    uint64_t *psync = (uint64_t *)nvshmem_malloc(sizeof(uint64_t) * npes);

    dim3 gridDim(1), blockDim(WARP_SIZE * 2);
    void *args[] = { &data, &psync, &data_len };

    nvshmem_barrier_all();
    nvshmemx_collective_launch((const void *)put_signal, gridDim, blockDim, args, 0, 0);
    nvshmem_barrier_all();

    nvshmem_free(data);
    free(psync_h);
    nvshmem_free(psync);

    nvshmem_finalize();
    return 0;
}