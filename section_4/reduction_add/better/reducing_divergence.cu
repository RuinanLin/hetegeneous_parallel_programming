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
float cpu_sum_reduction(float *input, int vector_size);
float gpu_sum_reduction(float *input, int vector_size);
__global__ void gpu_sum_reduction_kernel(float *d_input, float *d_output);


int main()
{
    // initialize the vector to be operated on
    int vector_size = 1073741824;
    float *input = (float *)malloc(vector_size * sizeof(float));
    if (input == NULL)
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
    float cpu_res = cpu_sum_reduction(input, vector_size);
    gettimeofday(&end, 0);
    elapsed_sec = (double)(end.tv_sec - begin.tv_sec) + (double)(end.tv_usec - begin.tv_usec) / 1000000.0;
    printf("cpu_res: %f\n", cpu_res);
    printf("cpu_time: %lf sec\n", elapsed_sec);
    
    // launch the function using GPU
    printf("GPU start ...\n");
    gettimeofday(&begin, NULL);
    float gpu_res = gpu_sum_reduction(input, vector_size);
    gettimeofday(&end, 0);
    elapsed_sec = (double)(end.tv_sec - begin.tv_sec) + (double)(end.tv_usec - begin.tv_usec) / 1000000.0;
    printf("gpu_res: %f\n", gpu_res);
    printf("gpu_time: %lf sec\n", elapsed_sec);

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

float gpu_sum_reduction(float *input, int vector_size)
{
    // we perform out-of-place reduction, so we need the place to store the temporary results
    float *temp = (float *)malloc((vector_size % (BLOCK_SIZE * 2) + vector_size / (BLOCK_SIZE * 2)) * sizeof(float));
    if (temp == NULL)
    {
        fprintf(stderr, "error: malloc() failed in file '%s' in line %i.\n", __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    // prepare the first vector_size%(BLOCK_SIZE*2) elements
    memcpy(temp, input, (vector_size % (BLOCK_SIZE * 2)) * sizeof(float));

    // parallel part
    int first_loop_flag = 1;
    float *input_pointer;
    while (vector_size >= BLOCK_SIZE * 2)
    {
        // decide the remaining element number and number of blocks in this loop
        int num_remaining = vector_size % (BLOCK_SIZE * 2);
        int num_blocks = vector_size / (BLOCK_SIZE * 2);

        // decide the position of 'input_pointer'
        if (first_loop_flag)
        {
            first_loop_flag = 0;
            input_pointer = input + num_remaining;
        }
        else
            input_pointer = temp + num_remaining;

        // allocate global memory on the GPU
        float *d_input;
        CUDA_SAFE_CALL(cudaMalloc((void **)&d_input, num_blocks * BLOCK_SIZE * 2 * sizeof(float)));
        CUDA_SAFE_CALL(cudaMemcpy(d_input, input_pointer, num_blocks * BLOCK_SIZE * 2 * sizeof(float), cudaMemcpyHostToDevice));
        float *d_output;
        CUDA_SAFE_CALL(cudaMalloc((void **)&d_output, num_blocks * sizeof(float)));

        // launch the kernel function
        gpu_sum_reduction_kernel<<<num_blocks, BLOCK_SIZE>>>(d_input, d_output);

        // copy the partial sum from GPU's global memory to temp
        CUDA_SAFE_CALL(cudaMemcpy(temp + num_remaining, d_output, num_blocks * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaFree(d_input));
        CUDA_SAFE_CALL(cudaFree(d_output));

        // update the vector_size
        vector_size = num_remaining + num_blocks;
    }

    // sequentially process the remainings
    float sum = cpu_sum_reduction(temp, vector_size);
    free(temp);
    return sum;
}

__global__
void gpu_sum_reduction_kernel(float *d_input, float *d_output)
{
    // calculate the global_thread_idx
    int block_input_start_idx = blockIdx.x * blockDim.x * 2;

    // load data from global memory to shared memory; each thread is responsible to two elements
    __shared__ float sh_input[BLOCK_SIZE * 2];
    sh_input[threadIdx.x] = d_input[block_input_start_idx + threadIdx.x];
    sh_input[threadIdx.x + BLOCK_SIZE] = d_input[block_input_start_idx + BLOCK_SIZE + threadIdx.x];

    // calculate
    for (int stride = BLOCK_SIZE; stride > 0; stride /= 2)
    {
        __syncthreads();
        if (threadIdx.x < stride)
            sh_input[threadIdx.x] += sh_input[threadIdx.x + stride];
    }

    // store the sum to the global memory
    if (threadIdx.x == 0)
        d_output[blockIdx.x] = sh_input[0];
}
