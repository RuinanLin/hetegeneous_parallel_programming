#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

#define VECTOR_SIZE 100000000
#define SEG_SIZE 65536

#define BLOCK_SIZE 256

#define LOW (-10.0)
#define HIGH 10.0

# define CUDA_SAFE_CALL(call) {                                                 \
    cudaError err = call;                                                       \
    if (cudaSuccess != err) {                                                   \
        fprintf(stderr, "error %d: Cuda error in file '%s' in line %i : %s.\n", \
                err, __FILE__, __LINE__, cudaGetErrorString(err));              \
        exit(EXIT_FAILURE);                                                     \
    }                                                                           \
}

float f_rand(float low, float high);
void init_vec(float *a, int size);
void vec_add_gpu(float *c, float *a, float *b, int vec_size);
__global__ void vec_add_kernel(float *c, float *a, float *b, int vec_size);
void vec_add_cpu(float *c, float *a, float *b, int vec_size);
void check_res(float *c_gpu, float *c_ref, float *a, float *b, int vec_size);

int main()
{
    // alloc memory for A, B, C_gpu, C_ref
    float *A;
    float *B;
    float *C_gpu;
    float *C_ref;
    CUDA_SAFE_CALL(cudaHostAlloc((void **)&A, VECTOR_SIZE * sizeof(float), cudaHostAllocDefault));
    CUDA_SAFE_CALL(cudaHostAlloc((void **)&B, VECTOR_SIZE * sizeof(float), cudaHostAllocDefault));
    CUDA_SAFE_CALL(cudaHostAlloc((void **)&C_gpu, VECTOR_SIZE * sizeof(float), cudaHostAllocDefault));
    CUDA_SAFE_CALL(cudaHostAlloc((void **)&C_ref, VECTOR_SIZE * sizeof(float), cudaHostAllocDefault));

    // initialize the vectors
    init_vec(A, VECTOR_SIZE);
    init_vec(B, VECTOR_SIZE);

    // prepare timer
    struct timeval begin, end;
    double elapsed_sec;

    // perform vector adding using serial multi-gpu
    printf("GPU start ...\n");
    gettimeofday(&begin, NULL);
    vec_add_gpu(C_gpu, A, B, VECTOR_SIZE);
    gettimeofday(&end, NULL);
    elapsed_sec = (double)(end.tv_sec - begin.tv_sec) + (double)(end.tv_usec - begin.tv_usec) / 1000000.0;
    printf("gpu_time: %lf sec\n", elapsed_sec);

    // perform vector adding using totally sequential cpu
    printf("CPU start ...\n");
    gettimeofday(&begin, NULL);
    vec_add_cpu(C_ref, A, B, VECTOR_SIZE);
    gettimeofday(&end, NULL);
    elapsed_sec = (double)(end.tv_sec - begin.tv_sec) + (double)(end.tv_usec - begin.tv_usec) / 1000000.0;
    printf("cpu_time: %lf sec\n", elapsed_sec);

    // check result
    check_res(C_gpu, C_ref, A, B, VECTOR_SIZE);

    CUDA_SAFE_CALL(cudaFreeHost(A));
    CUDA_SAFE_CALL(cudaFreeHost(B));
    CUDA_SAFE_CALL(cudaFreeHost(C_gpu));
    CUDA_SAFE_CALL(cudaFreeHost(C_ref));
    return 0;
}

float f_rand(float low, float high)
{
    return (float)rand() / (float)RAND_MAX * (high - low) + low;
}

void init_vec(float *a, int size)
{
    srand(time(NULL));
    for (int i = 0; i < size; i++)
        a[i] = f_rand(LOW, HIGH);
}

void vec_add_gpu(float *c, float *a, float *b, int vec_size)
{
    // create two streams
    cudaStream_t stream0, stream1;
    CUDA_SAFE_CALL(cudaStreamCreate(&stream0));
    CUDA_SAFE_CALL(cudaStreamCreate(&stream1));

    // alloc device memory
    float *d_a0, *d_b0, *d_c0;
    float *d_a1, *d_b1, *d_c1;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_a0, SEG_SIZE * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_b0, SEG_SIZE * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_c0, SEG_SIZE * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_a1, SEG_SIZE * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_b1, SEG_SIZE * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_c1, SEG_SIZE * sizeof(float)));

    // iterate through every segment
    for (int segment_start_idx = 0; segment_start_idx < vec_size; segment_start_idx += SEG_SIZE * 2)
    {
        // stream 0
        int seg_size_0 = (vec_size - segment_start_idx > SEG_SIZE) ? SEG_SIZE : (vec_size - segment_start_idx);
        CUDA_SAFE_CALL(cudaMemcpyAsync(d_a0, a + segment_start_idx, seg_size_0 * sizeof(float), cudaMemcpyHostToDevice, stream0));
        CUDA_SAFE_CALL(cudaMemcpyAsync(d_b0, b + segment_start_idx, seg_size_0 * sizeof(float), cudaMemcpyHostToDevice, stream0));
        vec_add_kernel<<<(seg_size_0 - 1) / BLOCK_SIZE + 1, BLOCK_SIZE, 0, stream0>>>(d_c0, d_a0, d_b0, seg_size_0);
        CUDA_SAFE_CALL(cudaMemcpyAsync(c + segment_start_idx, d_c0, seg_size_0 * sizeof(float), cudaMemcpyDeviceToHost, stream0));

        // stream 1
        if (vec_size - segment_start_idx > SEG_SIZE)
        {
            int seg_size_1 = (vec_size - segment_start_idx - SEG_SIZE > SEG_SIZE) ? SEG_SIZE : (vec_size - segment_start_idx - SEG_SIZE);
            CUDA_SAFE_CALL(cudaMemcpyAsync(d_a1, a + segment_start_idx + SEG_SIZE, seg_size_1 * sizeof(float), cudaMemcpyHostToDevice, stream1));
            CUDA_SAFE_CALL(cudaMemcpyAsync(d_b1, b + segment_start_idx + SEG_SIZE, seg_size_1 * sizeof(float), cudaMemcpyHostToDevice, stream1));
            vec_add_kernel<<<(seg_size_1 - 1) / BLOCK_SIZE + 1, BLOCK_SIZE, 0, stream1>>>(d_c1, d_a1, d_b1, seg_size_1);
            CUDA_SAFE_CALL(cudaMemcpyAsync(c + segment_start_idx + SEG_SIZE, d_c1, seg_size_1 * sizeof(float), cudaMemcpyDeviceToHost, stream1));
        }
    }

    // wait until all the tasks in all streams have completed
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    // free the device memory
    CUDA_SAFE_CALL(cudaFree(d_a0));
    CUDA_SAFE_CALL(cudaFree(d_b0));
    CUDA_SAFE_CALL(cudaFree(d_c0));
    CUDA_SAFE_CALL(cudaFree(d_a1));
    CUDA_SAFE_CALL(cudaFree(d_b1));
    CUDA_SAFE_CALL(cudaFree(d_c1));
}

__global__ void vec_add_kernel(float *c, float *a, float *b, int vec_size)
{
    // calc global thread index
    int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // add
    if (global_thread_idx < vec_size)
        c[global_thread_idx] = a[global_thread_idx] + b[global_thread_idx];
}

void vec_add_cpu(float *c, float *a, float *b, int vec_size)
{
    for (int i = 0; i < vec_size; i++)
        c[i] = a[i] + b[i];
}

void check_res(float *c_gpu, float *c_ref, float *a, float *b, int vec_size)
{
    for (int i = 0; i < vec_size; i++)
    {
        if (abs(c_gpu[i] - c_ref[i]) >= 0.01)
        {
            printf("INCORRECT!\na[%d] = %f, b[%d] = %f\nc_gpu[%d] = %f, c_ref[%d] = %f\n", i, a[i], i, b[i], i, c_gpu[i], i, c_ref[i]);
            return;
        }
    }
    printf("CORRECT!\n");
}
