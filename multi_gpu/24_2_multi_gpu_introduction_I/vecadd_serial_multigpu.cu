#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

#define VECTOR_SIZE 100000000

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

float A[VECTOR_SIZE];
float B[VECTOR_SIZE];
float C_gpu[VECTOR_SIZE];
float C_ref[VECTOR_SIZE];

float f_rand(float low, float high);
void init_vec(float *a, int size);
void vec_add_serial_multi_gpu(float *c, float *a, float *b, int vec_size);
__global__ void vec_add_kernel(float *c, float *a, float *b, int vec_size);
void vec_add_cpu(float *c, float *a, float *b, int vec_size);
void check_res(float *c_gpu, float *c_ref, float *a, float *b, int vec_size);

int main()
{
    // initialize the vectors
    init_vec(A, VECTOR_SIZE);
    init_vec(B, VECTOR_SIZE);

    // prepare timer
    struct timeval begin, end;
    double elapsed_sec;

    // perform vector adding using serial multi-gpu
    printf("GPU start ...\n");
    gettimeofday(&begin, NULL);
    vec_add_serial_multi_gpu(C_gpu, A, B, VECTOR_SIZE);
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

void vec_add_serial_multi_gpu(float *c, float *a, float *b, int vec_size)
{
    // get how many gpus do I have
    int gpu_count;
    CUDA_SAFE_CALL(cudaGetDeviceCount(&gpu_count));

    // how many elements should each gpu consider
    int normal_tile_size = (VECTOR_SIZE - 1) / gpu_count + 1;

    // declare device pointers
    float **d_a = (float **)malloc(gpu_count * sizeof(float *));
    float **d_b = (float **)malloc(gpu_count * sizeof(float *));
    float **d_c = (float **)malloc(gpu_count * sizeof(float *));

    // alloc memory in each device
    for (int device_idx = 0; device_idx < gpu_count; device_idx++)
    {
        // set a new active device
        CUDA_SAFE_CALL(cudaSetDevice(device_idx));
        printf("\tGPU %d malloc ...\n", device_idx);

        // allocate the global memory of the device
        int tile_size = (device_idx < gpu_count - 1) ? normal_tile_size : (vec_size - normal_tile_size * (gpu_count - 1));
        int vec_start_idx = device_idx * normal_tile_size;
        CUDA_SAFE_CALL(cudaMalloc((void **)&(d_a[device_idx]), tile_size * sizeof(float)));
        CUDA_SAFE_CALL(cudaMemcpy(d_a[device_idx], a + vec_start_idx, tile_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMalloc((void **)&(d_b[device_idx]), tile_size * sizeof(float)));
        CUDA_SAFE_CALL(cudaMemcpy(d_b[device_idx], b + vec_start_idx, tile_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMalloc((void **)&(d_c[device_idx]), tile_size * sizeof(float)));
    }

    // launch kernel in each device
    for (int device_idx = 0; device_idx < gpu_count; device_idx++)
    {
        // set a new active device
        CUDA_SAFE_CALL(cudaSetDevice(device_idx));
        printf("\tGPU %d start kernel ...\n", device_idx);

        // launch the kernel
        int tile_size = (device_idx < gpu_count - 1) ? normal_tile_size : (vec_size - normal_tile_size * (gpu_count - 1));
        vec_add_kernel<<<(tile_size - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(d_c[device_idx], d_a[device_idx], d_b[device_idx], tile_size);

        // free the adders' memory
        CUDA_SAFE_CALL(cudaFree(d_a[device_idx]));
        CUDA_SAFE_CALL(cudaFree(d_b[device_idx]));
    }

    // free d_a and d_b
    free(d_a);
    free(d_b);

    // copy result and free the memory in each device
    for (int device_idx = 0; device_idx < gpu_count; device_idx++)
    {
        // set a new active device
        CUDA_SAFE_CALL(cudaSetDevice(device_idx));
        printf("\tGPU %d copy result ...\n", device_idx);

        // copy
        int tile_size = (device_idx < gpu_count - 1) ? normal_tile_size : (vec_size - normal_tile_size * (gpu_count - 1));
        int vec_start_idx = device_idx * normal_tile_size;
        CUDA_SAFE_CALL(cudaMemcpy(c + vec_start_idx, d_c[device_idx], tile_size * sizeof(float), cudaMemcpyDeviceToHost));

        // free d_c[]
        CUDA_SAFE_CALL(cudaFree(d_c[device_idx]));
    }

    // free d_c
    free(d_c);
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
