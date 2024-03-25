#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

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
void cpu_scan(float *output, float *input, int vector_size);
void gpu_scan(float *output, float *input, int vector_size);
__global__ void gpu_scan_kernel(float *d_input, int vector_size, float *d_block_tail);
__global__ void gpu_broadcast_kernel(float *d_input, int vector_size, float *d_block_tail);


int main()
{
    // initialize the vector to be operated on
    int vector_size = 1073741824;
    float *input = (float *)malloc(vector_size * sizeof(float));
    float *output_cpu = (float *)malloc(vector_size * sizeof(float));
    float *output_gpu = (float *)malloc(vector_size * sizeof(float));
    if (input == NULL || output_cpu == NULL || output_gpu == NULL)
    {
        fprintf(stderr, "error: malloc() failed in file '%s' in line %i.\n", __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    srand(time(0));
    for (int i = 0; i < vector_size; i++)
        input[i] = float_rand(LOW, HIGH);

    // prepare timer
    struct timeval begin, end;
    double elapsed_sec;

    // launch the function using CPU
    printf("CPU start ...\n");
    gettimeofday(&begin, NULL);
    cpu_scan(output_cpu, input, vector_size);
    gettimeofday(&end, 0);
    elapsed_sec = (double)(end.tv_sec - begin.tv_sec) + (double)(end.tv_usec - begin.tv_usec) / 1000000.0;
    printf("cpu_time: %lf sec\n", elapsed_sec);
    
    // launch the function using GPU
    printf("GPU start ...\n");
    gettimeofday(&begin, NULL);
    gpu_scan(output_gpu, input, vector_size);
    gettimeofday(&end, 0);
    elapsed_sec = (double)(end.tv_sec - begin.tv_sec) + (double)(end.tv_usec - begin.tv_usec) / 1000000.0;
    printf("gpu_time: %lf sec\n", elapsed_sec);

    // result checking
    for (int i = 0; i < vector_size; i++)
    {
        if (abs(output_cpu[i] - output_gpu[i]) >= 12.0)
        {
            printf("INCORRECT!\n");
            printf("output_cpu[%d] = %f, ", i, output_cpu[i]);
            printf("output_gpu[%d] = %f\n", i, output_gpu[i]);
            return 0;
        }
    }
    printf("CORRECT!\n");

    free(input);
    free(output_cpu);
    free(output_gpu);
    return 0;
}

float float_rand(float high, float low)
{
    return (float)rand() / (float)RAND_MAX * (high - low) + low;
}

void cpu_scan(float *output, float *input, int vector_size)
{
    output[0] = input[0];
    for (int i = 1; i < vector_size; i++)
        output[i] = output[i - 1] + input[i];
}

void gpu_scan(float *output, float *input, int vector_size)
{
    // allocate input space on the device
    float *d_input;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_input, vector_size * sizeof(float)));
    CUDA_SAFE_CALL(cudaMemcpy(d_input, input, vector_size * sizeof(float), cudaMemcpyHostToDevice));

    // launch the kernel
    int num_blocks = (vector_size - 1) / (BLOCK_SIZE * 2) + 1;
    if (num_blocks == 1)
        gpu_scan_kernel<<<num_blocks, BLOCK_SIZE>>>(d_input, vector_size, NULL);
    else
    {
        // allocate space to store the tails of each block
        float *h_block_tail = (float *)malloc((num_blocks - 1) * sizeof(float));
        float *d_block_tail;
        CUDA_SAFE_CALL(cudaMalloc((void **)&d_block_tail, (num_blocks - 1) * sizeof(float)));

        // launch the kernel
        gpu_scan_kernel<<<num_blocks, BLOCK_SIZE>>>(d_input, vector_size, d_block_tail);

        // gather the tail result
        CUDA_SAFE_CALL(cudaMemcpy(h_block_tail, d_block_tail, (num_blocks - 1) * sizeof(float), cudaMemcpyDeviceToHost));

        // recursively call the gpu_scan() on the block_tail array
        gpu_scan(h_block_tail, h_block_tail, num_blocks - 1);

        // broadcast the tail results to all the blocks
        CUDA_SAFE_CALL(cudaMemcpy(d_block_tail, h_block_tail, (num_blocks - 1) * sizeof(float), cudaMemcpyHostToDevice));
        gpu_broadcast_kernel<<<(vector_size - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(d_input, vector_size, d_block_tail);

        free(h_block_tail);
        CUDA_SAFE_CALL(cudaFree(d_block_tail));
    }
    CUDA_SAFE_CALL(cudaMemcpy(output, d_input, vector_size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaFree(d_input));
}

__global__
void gpu_scan_kernel(float *d_input, int vector_size, float *d_block_tail)
{
    // calculate the scan_index for each individual thread
    int global_thread_index_start = blockIdx.x * blockDim.x;

    // load the input into the shared memory
    __shared__ float block_input[BLOCK_SIZE * 2];  
    int responsible_index_1 = global_thread_index_start * 2 + threadIdx.x;
    int responsible_index_2 = global_thread_index_start * 2 + BLOCK_SIZE + threadIdx.x;
    if (responsible_index_1 < vector_size)
        block_input[threadIdx.x] = d_input[global_thread_index_start * 2 + threadIdx.x];
    if (responsible_index_2 < vector_size)
        block_input[BLOCK_SIZE + threadIdx.x] = d_input[global_thread_index_start * 2 + BLOCK_SIZE + threadIdx.x];

    // reduction phase
    for (int stride = 1; stride < 2 * BLOCK_SIZE; stride *= 2)
    {
        __syncthreads();
        int processing_index = (threadIdx.x + 1) * (stride * 2) - 1;
        if (processing_index < 2 * BLOCK_SIZE && global_thread_index_start + processing_index < vector_size)
            block_input[processing_index] += block_input[processing_index - stride];
    }

    // post reduction reverse phase
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2)
    {
        __syncthreads();
        int processing_index = (threadIdx.x + 1) * (stride * 2) - 1;
        if (processing_index + stride < 2 * BLOCK_SIZE && global_thread_index_start + processing_index + stride < vector_size)
            block_input[processing_index + stride] += block_input[processing_index];
    }
    __syncthreads();

    // store the result to global memory
    if (responsible_index_1 < vector_size)
        d_input[global_thread_index_start * 2 + threadIdx.x] = block_input[threadIdx.x];
    if (responsible_index_2 < vector_size)
        d_input[global_thread_index_start * 2 + BLOCK_SIZE + threadIdx.x] = block_input[BLOCK_SIZE + threadIdx.x];

    // store the tail to d_block_tail[]
    int num_blocks = (vector_size - 1) / (2 * BLOCK_SIZE) + 1;
    if (blockIdx.x < num_blocks - 1 && threadIdx.x == 0)
        d_block_tail[blockIdx.x] = block_input[BLOCK_SIZE * 2 - 1];
}

__global__
void gpu_broadcast_kernel(float *d_input, int vector_size, float *d_block_tail)
{
    // calculate the global_index for each thread
    int global_index = blockIdx.x * blockDim.x + threadIdx.x;

    // broadcast
    if (global_index < vector_size && global_index / (BLOCK_SIZE * 2) > 0)
        d_input[global_index] += d_block_tail[global_index / (BLOCK_SIZE * 2) - 1];
}
