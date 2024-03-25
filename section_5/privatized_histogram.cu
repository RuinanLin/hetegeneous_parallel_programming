#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#define CHAR_RANGE 128

#define TEXT_LEN 1000000
#define PATH "./text"

#define NUM_BLOCK 1024
#define BLOCK_SIZE 256

# define CUDA_SAFE_CALL(call) {                                                 \
    cudaError err = call;                                                       \
    if (cudaSuccess != err) {                                                   \
        fprintf(stderr, "error %d: Cuda error in file '%s' in line %i : %s.\n", \
                err, __FILE__, __LINE__, cudaGetErrorString(err));              \
        exit(EXIT_FAILURE);                                                     \
    }                                                                           \
}

// allocate the output space for both cpu and gpu
int cpu_output[CHAR_RANGE];
int gpu_output[CHAR_RANGE];

FILE *create_text(const char *file_name);
void cpu_histogram(int *cpu_output, char *text, int text_len);
void gpu_histogram(int *gpu_output, char *text, int text_len);
__global__ void gpu_histogram_kernel(int *d_output, char *d_text, int text_len);

int main()
{
    // create a text file if it doesn't exist, and then read the text file
    FILE *fp = create_text(PATH);
    char *text = (char *)malloc(TEXT_LEN * sizeof(char));
    if (text == NULL)
    {
        fprintf(stderr, "error: malloc fails in file '%s' in line %i\n", __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    fgets(text, TEXT_LEN, fp);
    fclose(fp);

    // prepare timer
    struct timeval begin, end;
    double elapsed_sec;

    // launch the function using CPU
    printf("CPU start ...\n");
    gettimeofday(&begin, NULL);
    cpu_histogram(cpu_output, text, TEXT_LEN);
    gettimeofday(&end, NULL);
    elapsed_sec = (double)(end.tv_sec - begin.tv_sec) + (double)(end.tv_usec - begin.tv_usec) / 1000000.0;
    printf("cpu_time: %lf sec\n", elapsed_sec);

    // launch the function using GPU
    printf("GPU start ...\n");
    gettimeofday(&begin, NULL);
    gpu_histogram(gpu_output, text, TEXT_LEN);
    gettimeofday(&end, NULL);
    elapsed_sec = (double)(end.tv_sec - begin.tv_sec) + (double)(end.tv_usec - begin.tv_usec) / 1000000.0;
    printf("gpu_time: %lf sec\n", elapsed_sec);

    // result checking
    for (int i = 0; i < CHAR_RANGE; i++)
    {
        if (cpu_output[i] != gpu_output[i])
        {
            printf("INCORRECT!\n");
            printf("cpu_output[%d] = %d, gpu_output[%d] = %d\n", i, cpu_output[i], i, gpu_output[i]);
            return 0;
        }
    }
    printf("CORRECT!\n");

    free(text);
    return 0;
}

FILE *create_text(const char *file_name)
{
    // try to open the file
    FILE *fp = fopen(file_name, "r");

    // if the file does not exist, then create this file
    if (fp == NULL)
    {
        fp = fopen(file_name, "w");
        if (fp == NULL)
        {
            fprintf(stderr, "error: Cannot create text file in file '%s' in line %i\n", __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        }
        char *text = (char *)malloc((TEXT_LEN + 1) * sizeof(char));
        if (text == NULL)
        {
            fprintf(stderr, "error: malloc fails in file '%s' in line %i\n", __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        }
        srand(time(NULL));
        for (int i = 0; i < TEXT_LEN; i++)
            text[i] = rand() % CHAR_RANGE;
        text[TEXT_LEN] = '\0';
        fprintf(fp, text);
        free(text);
        fclose(fp);
        fp = fopen(file_name, "r");
    }

    // return the opened file pointer
    return fp;
}

void cpu_histogram(int *cpu_output, char *text, int text_len)
{
    for (int i = 0; i < text_len; i++)
        cpu_output[text[i]]++;
}

void gpu_histogram(int *gpu_output, char *text, int text_len)
{
    // allocate global memory in gpu
    char *d_text;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_text, text_len * sizeof(char)));
    CUDA_SAFE_CALL(cudaMemcpy(d_text, text, text_len * sizeof(char), cudaMemcpyHostToDevice));
    int *d_output;
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_output, CHAR_RANGE * sizeof(int)));

    // launch the kernel function
    gpu_histogram_kernel<<<NUM_BLOCK, BLOCK_SIZE>>>(d_output, d_text, text_len);

    // copy the result out
    CUDA_SAFE_CALL(cudaMemcpy(gpu_output, d_output, CHAR_RANGE * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaFree(d_text));
    CUDA_SAFE_CALL(cudaFree(d_output));
}

__global__ void gpu_histogram_kernel(int *d_output, char *d_text, int text_len)
{
    // allocate the shared memory
    __shared__ int histo_private[CHAR_RANGE];

    // clear the histo_private[]
    for (int i = threadIdx.x; i < CHAR_RANGE; i += blockDim.x)
        histo_private[i] = 0;

    // global thread index
    int global_thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    // calculate the histogram
    for (int i = global_thread_index; i < text_len; i += total_threads)
        atomicAdd(histo_private + d_text[i], 1);
    __syncthreads();

    // store the result to global memory
    for (int i = threadIdx.x; i < CHAR_RANGE; i += blockDim.x)
        atomicAdd(d_output + i, histo_private[i]);
}