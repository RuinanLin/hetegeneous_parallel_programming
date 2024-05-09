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

#define NVSHMEM_CHECK(stmt)                                                                \
    do {                                                                                   \
        int result = (stmt);                                                               \
        if (NVSHMEMX_SUCCESS != result) {                                                  \
            fprintf(stderr, "[%s:%d] nvshmem failed with error %d \n", __FILE__, __LINE__, \
                    result);                                                               \
            exit(-1);                                                                      \
        }                                                                                  \
    } while (0)

#define WARP_SIZE   32
#define QUEUE_SIZE  8

__global__ void remote_add_test(int *count, int *queue, int *head, int *tail, int *tail_lock, int *n_finish) {
    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();

    // about the threads and blocks
    int thread_lane = threadIdx.x & (WARP_SIZE-1);          // thread index within the warp
    int warp_lane = threadIdx.x / WARP_SIZE;                // warp index within the CTA

    if (thread_lane == 0) {
        if (warp_lane == 2) {
            printf("PE[%d] : lane 2 start.\n", mype);
            int my_n_finish;
            do {
                int my_head = *head;
                printf("PE[%d] : my_head = %d\n", mype, my_head);
                int my_tail;
                do {
                    my_tail = *tail;
                    // printf("PE[%d] : my_tail = %d\n", mype, my_tail);
                } while (my_head == my_tail);
                printf("PE[%d] get message. head = %d, tail = %d, *head == *tail : %d\n", mype, my_head, my_tail, my_head == my_tail);
                *count += queue[my_head];
                // *head = (*head + 1) % QUEUE_SIZE;
                nvshmem_int_p(head, (my_head + 1) % QUEUE_SIZE, mype);
                nvshmem_quiet();
                printf("PE[%d] : head added to %d\n", mype, *head);

                my_n_finish = *n_finish;
            } while (my_n_finish < (npes - 1) * 2);
        } else {
            printf("PE[%d] : lane %d start.\n", mype, warp_lane);
            for (int msg = 1; msg <= 2; msg++) {
                for (int recv = 0; recv < npes; recv++) {
                    if (recv == mype)
                        continue;
                    while (nvshmem_int_atomic_swap(tail_lock, 1, recv))
                        ;
                    printf("PE[%d] lane[%d] aquired lock[%d]! %d\n", mype, warp_lane, recv, (nvshmem_int_g(head, recv) + QUEUE_SIZE - nvshmem_int_g(tail, recv)) % QUEUE_SIZE);
                    while ((nvshmem_int_g(head, recv) + QUEUE_SIZE - nvshmem_int_g(tail, recv)) % QUEUE_SIZE == 1) {
                        // printf("head = %d\n", nvshmem_int_g(head, recv));
                    }
                    printf("PE[%d] lane[%d] have empty space to write in [%d].\n", mype, warp_lane, recv);
                    int recv_tail = nvshmem_int_g(tail, recv);
                    nvshmem_int_p(queue + recv_tail, msg, recv);
                    nvshmem_int_p(tail, (recv_tail + 1) % QUEUE_SIZE, recv);
                    nvshmem_quiet();
                    nvshmem_int_p(tail_lock, 0, recv);
                    nvshmem_quiet();
                }
            }
            for (int recv = 0; recv < npes; recv++) {
                if (recv == mype)
                    continue;
                nvshmem_int_atomic_inc(n_finish, recv);
                nvshmem_quiet();
                printf("n_finish = %d\n", *n_finish);
            }
        }
    }
}

int main(int c, char *v[]) {
    int mype, npes, mype_node;

    nvshmem_init();
    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();
    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);

    // application picks the device each PE will use
    CUDA_SAFE_CALL(cudaSetDevice(mype_node));

    // allocate memory
    int *d_queue = (int *)nvshmem_malloc(QUEUE_SIZE * sizeof(int));

    int h_head = 0;
    int *d_head = (int *)nvshmem_malloc(sizeof(int));
    CUDA_SAFE_CALL(cudaMemcpy(d_head, &h_head, sizeof(int), cudaMemcpyHostToDevice));

    int h_tail = 0;
    int *d_tail = (int *)nvshmem_malloc(sizeof(int));
    CUDA_SAFE_CALL(cudaMemcpy(d_tail, &h_tail, sizeof(int), cudaMemcpyHostToDevice));

    int h_tail_lock = 0;
    int *d_tail_lock = (int *)nvshmem_malloc(sizeof(int));
    CUDA_SAFE_CALL(cudaMemcpy(d_tail_lock, &h_tail_lock, sizeof(int), cudaMemcpyHostToDevice));

    int h_n_finish = 0;
    int *d_n_finish = (int *)nvshmem_malloc(sizeof(int));
    CUDA_SAFE_CALL(cudaMemcpy(d_n_finish, &h_n_finish, sizeof(int), cudaMemcpyHostToDevice));

    int h_count = 0;
    int *d_count;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_count, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpy(d_count, &h_count, sizeof(int), cudaMemcpyHostToDevice));

    // launch the kernel
    std::cout << "PE[" << mype << "] launched.\n";
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    nvshmem_barrier_all();
    remote_add_test<<<1, 3*WARP_SIZE>>>(d_count, d_queue, d_head, d_tail, d_tail_lock, d_n_finish);

    // print the result
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    nvshmem_barrier_all();
    CUDA_SAFE_CALL(cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "PE[" << mype << "] : " << h_count << "\n";
    return 0;
}