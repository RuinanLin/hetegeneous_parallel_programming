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

float f_rand(float low, float high);
void init_vec(float *a, int size);
void vec_add_gpu(float *c, float *a, float *b, int vec_size);
__global__ void vec_add_kernel(float *c, float *a, float *b, int vec_size);
void vec_add_cpu(float *c, float *a, float *b, int vec_size);
void check_res(float *c_gpu, float *c_ref, float *a, float *b, int vec_size);

int main()
{
    // alloc memory for A, B, C_gpu, C_ref
    float *A = (float *)malloc(VECTOR_SIZE * sizeof(float));
    float *B = (float *)malloc(VECTOR_SIZE * sizeof(float));
    float *C_gpu = (float *)malloc(VECTOR_SIZE * sizeof(float));
    float *C_ref = (float *)malloc(VECTOR_SIZE * sizeof(float));

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

    free(A);
    free(B);
    free(C_gpu);
    free(C_ref);
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
    // alloc device memory
    float *d_a;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_a, vec_size * sizeof(float)));
    CUDA_SAFE_CALL(cudaMemcpy(d_a, a, vec_size * sizeof(float), cudaMemcpyHostToDevice));
    float *d_b;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_b, vec_size * sizeof(float)));
    CUDA_SAFE_CALL(cudaMemcpy(d_b, b, vec_size * sizeof(float), cudaMemcpyHostToDevice));
    float *d_c;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_c, vec_size * sizeof(float)));

    // launch kernel function
    vec_add_kernel<<<(vec_size - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(d_c, d_a, d_b, vec_size);

    // copy the result out
    CUDA_SAFE_CALL(cudaMemcpy(c, d_c, vec_size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaFree(d_a));
    CUDA_SAFE_CALL(cudaFree(d_b));
    CUDA_SAFE_CALL(cudaFree(d_c));
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
