#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define BLOCK_SIZE 256

#define LOW (-1.0)
#define HIGH 1.0

# define CUDA_SAFE_CALL(call) {                                                 \
    cudaError err = call;                                                       \
    if (cudaSuccess != err) {                                                   \
        fprintf(stderr, "error %d: Cuda error in file '%s' in line %i : %s.\n", \
                err, __FILE__, __LINE__, cudaGetErrorString(err));              \
        exit(EXIT_FAILURE);                                                     \
    }                                                                           \
}

float float_rand(float high, float low);
float cpu_sum_reduction(float *input, int vector_size);
float gpu_sum_reduction(float *h_input, int vector_size);
__global__ void gpu_sum_reduction_kernel(float *d_input, float *d_sum);


int main()
{
    // initialize the vector to be operated on
    int vector_size = BLOCK_SIZE * 2;
    float *input = (float *)malloc(vector_size * sizeof(float));
    srand(time(0));
    for (int i = 0; i < vector_size; i++)
        input[i] = float_rand(LOW, HIGH);

    // launch the function using CPU
    float cpu_res = cpu_sum_reduction(input, vector_size);
    printf("cpu_res: %f\n", cpu_res);
    
    // launch the function using GPU
    float gpu_res = gpu_sum_reduction(input, vector_size);
    printf("gpu_res: %f\n", gpu_res);

    // judge the result
    if (abs(cpu_res - gpu_res) < 0.01)
        printf("CORRECT!\n");
    else
        printf("INCORRECT!\n");

    free(input);
    return 0;
}

float float_rand(float high, float low)
{
    return (float)rand() / (float)RAND_MAX * (high - low) + low;
}

float cpu_sum_reduction(float *input, int vector_size)
{
    float res = 0;
    for (int i = 0; i < vector_size; i++)
        res += input[i];
    return res;
}

float gpu_sum_reduction(float *h_input, int vector_size)
{
    // allocate global memory on the GPU
    float *d_input;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_input, vector_size * sizeof(float)));
    CUDA_SAFE_CALL(cudaMemcpy(d_input, h_input, vector_size * sizeof(float), cudaMemcpyHostToDevice));
    float *d_sum;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_sum, sizeof(float)));

    // launch the kernel function
    gpu_sum_reduction_kernel<<<1, BLOCK_SIZE>>>(d_input, d_sum);

    // read the result out from the global memory of GPU
    float h_sum;
    CUDA_SAFE_CALL(cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaFree(d_input));
    CUDA_SAFE_CALL(cudaFree(d_sum));
    return h_sum;
}

__global__
void gpu_sum_reduction_kernel(float *d_input, float *d_sum)
{
    // load data from global memory to shared memory; each thread is responsible to two elements
    __shared__ float sh_input[BLOCK_SIZE * 2];
    sh_input[threadIdx.x] = d_input[threadIdx.x];
    sh_input[threadIdx.x + BLOCK_SIZE] = d_input[threadIdx.x + BLOCK_SIZE];

    // calculate
    for (int stride = 1; stride <= BLOCK_SIZE; stride *= 2)
    {
        __syncthreads();
        if (threadIdx.x % stride == 0)
            sh_input[threadIdx.x * 2] += sh_input[threadIdx.x * 2 + stride];
    }

    // store the sum to the global memory
    if (threadIdx.x == 0)
        *d_sum = sh_input[0];
}
