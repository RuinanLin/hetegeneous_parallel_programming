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
void vec_add_explicit_transfer_multi_gpu(float *c, float *a, float *b, int vec_size);
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
    vec_add_explicit_transfer_multi_gpu(C_gpu, A, B, VECTOR_SIZE);
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

void vec_add_explicit_transfer_multi_gpu(float *c, float *a, float *b, int vec_size)
{
    // get how many gpus do I have
    int gpu_count;
    CUDA_SAFE_CALL(cudaGetDeviceCount(&gpu_count));

    // how many elements should each gpu consider
    int normal_tile_size = (VECTOR_SIZE - 1) / (gpu_count - 1) + 1;

    // declare device pointers
    float **d_a = (float **)malloc((gpu_count - 1) * sizeof(float *));
    float **d_b = (float **)malloc((gpu_count - 1) * sizeof(float *));

    // alloc memory in each device
    for (int device_idx = 0; device_idx < gpu_count - 1; device_idx++)
    {
        // set a new active device
        CUDA_SAFE_CALL(cudaSetDevice(device_idx));
        printf("\tGPU %d malloc ...\n", device_idx);

        // allocate the global memory of the device
        int tile_size = (device_idx < gpu_count - 2) ? normal_tile_size : (vec_size - normal_tile_size * (gpu_count - 2));
        int vec_start_idx = device_idx * normal_tile_size;
        CUDA_SAFE_CALL(cudaMalloc((void **)&(d_a[device_idx]), tile_size * sizeof(float)));
        CUDA_SAFE_CALL(cudaMemcpy(d_a[device_idx], a + vec_start_idx, tile_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMalloc((void **)&(d_b[device_idx]), tile_size * sizeof(float)));
        CUDA_SAFE_CALL(cudaMemcpy(d_b[device_idx], b + vec_start_idx, tile_size * sizeof(float), cudaMemcpyHostToDevice));
    }

    // now I regret: I want all the work done by the last gpu
    CUDA_SAFE_CALL(cudaSetDevice(gpu_count - 1));
    printf("\tGPU %d set ...\n", gpu_count - 1);
    float *d_a_last_gpu;
    float *d_b_last_gpu;
    float *d_c_last_gpu;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_a_last_gpu, vec_size * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_b_last_gpu, vec_size * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_c_last_gpu, vec_size * sizeof(float)));

    // set streams and peer transfer
    cudaStream_t *stream = (cudaStream_t *)malloc(((gpu_count - 1) * sizeof(cudaStream_t)));
    for (int device_idx = 0; device_idx < gpu_count - 1; device_idx++)
    {
        CUDA_SAFE_CALL(cudaSetDevice(device_idx));
        CUDA_SAFE_CALL(cudaStreamCreate(&(stream[device_idx])));
        int tile_size = (device_idx < gpu_count - 2) ? normal_tile_size : (vec_size - normal_tile_size * (gpu_count - 2));
        int vec_start_idx = device_idx * normal_tile_size;
        CUDA_SAFE_CALL(cudaMemcpyPeerAsync(d_a_last_gpu + vec_start_idx, gpu_count - 1, d_a[device_idx], device_idx, tile_size * sizeof(float), stream[device_idx]));
        CUDA_SAFE_CALL(cudaMemcpyPeerAsync(d_b_last_gpu + vec_start_idx, gpu_count - 1, d_b[device_idx], device_idx, tile_size * sizeof(float), stream[device_idx]));
    }
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    // free the devices' memory
    for (int device_idx = 0; device_idx < gpu_count - 1; device_idx++)
    {
        CUDA_SAFE_CALL(cudaFree(d_a[device_idx]));
        CUDA_SAFE_CALL(cudaFree(d_b[device_idx]));
    }
    free(d_a);
    free(d_b);

    // set the last gpu to calc
    CUDA_SAFE_CALL(cudaSetDevice(gpu_count - 1));
    cudaStream_t stream_last_gpu;
    CUDA_SAFE_CALL(cudaStreamCreate(&stream_last_gpu));
    vec_add_kernel<<<(vec_size - 1) / BLOCK_SIZE + 1, BLOCK_SIZE, 0, stream_last_gpu>>>(d_c_last_gpu, d_a_last_gpu, d_b_last_gpu, vec_size);

    // copy the result out
    CUDA_SAFE_CALL(cudaMemcpyAsync(c, d_c_last_gpu, vec_size * sizeof(float), cudaMemcpyDeviceToHost, stream_last_gpu));
    CUDA_SAFE_CALL(cudaStreamSynchronize(stream_last_gpu));

    // free the last gpu's memory
    CUDA_SAFE_CALL(cudaFree(d_a_last_gpu));
    CUDA_SAFE_CALL(cudaFree(d_b_last_gpu));
    CUDA_SAFE_CALL(cudaFree(d_c_last_gpu));
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
