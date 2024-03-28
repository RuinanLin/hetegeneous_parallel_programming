// The performance is bad
// Because we have not fully utilized the GPU resource

#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <vector>
using namespace std;

#define VECTOR_SIZE 1000000000

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
    // get device count
    int device_count;
    CUDA_SAFE_CALL(cudaGetDeviceCount(&device_count));
    int normal_device_vec_size = (vec_size - 1) / device_count + 1;

    // create stream and allocate memory for each device
    std::vector<cudaStream_t> streams(device_count);
    std::vector<float *> d_a(device_count);
    std::vector<float *> d_b(device_count);
    std::vector<float *> d_c(device_count);
    std::vector<int> device_vec_size(device_count);
    std::vector<int> device_vec_start_idx(device_count);
    #pragma omp parallel for num_threads(device_count)
    for (int device_idx = 0; device_idx < device_count; device_idx++)
    {
        // set device and create stream
        CUDA_SAFE_CALL(cudaSetDevice(device_idx));
        CUDA_SAFE_CALL(cudaStreamCreate(&streams[device_idx]));

        // allocate and copy memory
        device_vec_size[device_idx] = (device_idx == device_count - 1) ? (vec_size - normal_device_vec_size * (device_count - 1)) : normal_device_vec_size;
        device_vec_start_idx[device_idx] = normal_device_vec_size * device_idx;
        CUDA_SAFE_CALL(cudaMalloc((void **)&d_a[device_idx], device_vec_size[device_idx] * sizeof(float)));
        CUDA_SAFE_CALL(cudaMemcpyAsync(d_a[device_idx], a + device_vec_start_idx[device_idx], device_vec_size[device_idx] * sizeof(float), cudaMemcpyHostToDevice, streams[device_idx]));
        CUDA_SAFE_CALL(cudaMalloc((void **)&d_b[device_idx], device_vec_size[device_idx] * sizeof(float)));
        CUDA_SAFE_CALL(cudaMemcpyAsync(d_b[device_idx], b + device_vec_start_idx[device_idx], device_vec_size[device_idx] * sizeof(float), cudaMemcpyHostToDevice, streams[device_idx]));
        CUDA_SAFE_CALL(cudaMalloc((void **)&d_c[device_idx], device_vec_size[device_idx] * sizeof(float)));
    }

    // launch kernels
    #pragma omp parallel for num_threads(device_count)
    for (int device_idx = 0; device_idx < device_count; device_idx++)
    {
        CUDA_SAFE_CALL(cudaSetDevice(device_idx));
        printf("\tDevice %d starts ...\n", device_idx);
        vec_add_kernel<<<(device_vec_size[device_idx] - 1) / BLOCK_SIZE + 1, BLOCK_SIZE, 0, streams[device_idx]>>>(d_c[device_idx], d_a[device_idx], d_b[device_idx], device_vec_size[device_idx]);
    }

    // copy results and free
    #pragma omp parallel for num_threads(device_count)
    for (int device_idx = 0; device_idx < device_count; device_idx++)
    {
        // copy
        CUDA_SAFE_CALL(cudaSetDevice(device_idx));
        CUDA_SAFE_CALL(cudaMemcpyAsync(c + device_vec_start_idx[device_idx], d_c[device_idx], device_vec_size[device_idx] * sizeof(float), cudaMemcpyDeviceToHost, streams[device_idx]));
        // CUDA_SAFE_CALL(cudaStreamSynchronize(streams[device_idx]));

        // free
        CUDA_SAFE_CALL(cudaFree(d_a[device_idx]));
        CUDA_SAFE_CALL(cudaFree(d_b[device_idx]));
        CUDA_SAFE_CALL(cudaFree(d_c[device_idx]));
    }
    // CUDA_SAFE_CALL(cudaDeviceSynchronize());
}

__global__ void vec_add_kernel(float *c, float *a, float *b, int vec_size)
{
    // calc global thread index
    int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // add
    if (global_thread_idx < vec_size)
        c[global_thread_idx] = a[global_thread_idx] * b[global_thread_idx];
}

void vec_add_cpu(float *c, float *a, float *b, int vec_size)
{
    for (int i = 0; i < vec_size; i++)
        c[i] = a[i] * b[i];
}

void check_res(float *c_gpu, float *c_ref, float *a, float *b, int vec_size)
{
    for (int i = 0; i < vec_size; i++)
    {
        if (abs(c_gpu[i] - c_ref[i]) >= 0.1)
        {
            printf("INCORRECT!\na[%d] = %f, b[%d] = %f\nc_gpu[%d] = %f, c_ref[%d] = %f\n", i, a[i], i, b[i], i, c_gpu[i], i, c_ref[i]);
            return;
        }
    }
    printf("CORRECT!\n");
}
