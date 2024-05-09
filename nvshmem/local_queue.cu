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
#define DATA_SIZE   1000

typedef struct {
    int write_finished;
    int data;
} message_t;

typedef struct {
    message_t messages[QUEUE_SIZE];
    int head;
    int tail;
    int lock;
} queue_t;

__global__ void remote_add_test(int *count, queue_t *queues) {
    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();

    // about the threads and blocks
    int thread_lane = threadIdx.x & (WARP_SIZE-1);          // thread index within the warp
    int warp_lane = threadIdx.x / WARP_SIZE;                // warp index within the CTA

    if (thread_lane == 0) {
        if (warp_lane == 2) {
            int message_count = 0;
            while (message_count < 6000) {
                for (int sender = 0; sender < npes; sender++) {
                    if (sender == mype) continue;
                    int queue_id = (sender < mype) ? mype - 1 : mype;
                    for (int message_id = 0; message_id < QUEUE_SIZE; message_id++) {
                        if (nvshmem_int_g(&(queues[queue_id].messages[message_id].write_finished), sender)) {
                            
                        }
                    }
                }
            }
        }
    }
}

void init_queues(queue_t *queue_vec, int n_queues) {
    for (int queue_id = 0; queue_id < n_queues; queue_id++) {
        queue_vec[queue_id].head = 0;
        queue_vec[queue_id].tail = 0;
        queue_vec[queue_id].lock = 0;
        for (int message_id = 0; message_id < QUEUE_SIZE; message_id++) {
            queue_vec[queue_id].messages[message_id].write_finished = 0;
            queue_vec[queue_id].messages[message_id].data_offset = 0;
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
    queue_t *h_queues = (queue_t *)malloc((npes - 1) * sizeof(queue_t));
    init_queues(h_queues, npes - 1);
    queue_t *d_queues = (queue_t *)nvshmem_malloc((npes - 1) * sizeof(queue_t));
    CUDA_SAFE_CALL(cudaMemcpy(d_queues, h_queues, (npes - 1) * sizeof(queue_t), cudaMemcpyHostToDevice));

    // launch kernel
    int h_count = 0;
    int *d_count;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_count, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpy(d_count, &h_count, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    nvshmem_barrier_all();
    remote_add_test<<<1, 3*WARP_SIZE>>>(d_count, d_queues);

    // print the result
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    nvshmem_barrier_all();
    CUDA_SAFE_CALL(cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "PE[" << mype << "] : " << h_count << "\n";
    return 0;
}