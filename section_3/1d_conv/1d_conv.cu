#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define WIDTH   10000000

#define LOW (-10.0)
#define HIGH 10.0

#define BLOCK_WIDTH 1024
#define MASK_WIDTH 7
#define O_TILE_WIDTH (BLOCK_WIDTH - MASK_WIDTH + 1)


float N[WIDTH];         // input
float M[MASK_WIDTH];    // mask
float P[WIDTH];         // output

float d_rand(float low, float high)
{
    return (float)rand() / (float)RAND_MAX * (high - low) + low;
}

void init()
{
    // initialize input data
    for (int i = 0; i < WIDTH; i++)
        N[i] = d_rand(LOW, HIGH);

    // initialize mask data
    for (int i = 0; i < MASK_WIDTH; i++)
        M[i] = d_rand(LOW, HIGH);
}

__global__
void cuda_conv_1d_kernel(float *d_P, float *d_N, const float * __restrict__ d_M)
{
    // allocate shared memory space
    __shared__ float Ns[BLOCK_WIDTH];

    // calculate who I am
    int index_o = blockIdx.x * O_TILE_WIDTH + threadIdx.x;
    int index_i = index_o - (MASK_WIDTH - 1) / 2;
    int tx = threadIdx.x;

    // load input data
    if (index_i >= 0 && index_i < WIDTH)
        Ns[tx] = d_N[index_i];
    else
        Ns[tx] = 0.0;
    __syncthreads();

    // calculate
    if (threadIdx.x < O_TILE_WIDTH)
    {
        float output = 0.0;
        for (int j = 0; j < MASK_WIDTH; j++)
            output += Ns[tx + j] * d_M[j];
        if (index_o < WIDTH)
            d_P[index_o] = output;
    }
}

void cuda_conv_1d(float *h_P, float *h_N, float *h_M)
{
    // allocate the space and copy the data to the GPU
    float *d_P;
    float *d_N;
    float *d_M;
    if (cudaMalloc((void **)&d_P, WIDTH * sizeof(float)) != cudaSuccess)
        printf("cudaMalloc not successful!\n");
    cudaMalloc((void **)&d_N, WIDTH * sizeof(float));
    cudaMemcpy(d_N, h_N, WIDTH * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_M, MASK_WIDTH * sizeof(float));
    cudaMemcpy(d_M, h_M, MASK_WIDTH * sizeof(float), cudaMemcpyHostToDevice);

    // call the kernel function
    cuda_conv_1d_kernel<<<(WIDTH - 1) / O_TILE_WIDTH + 1, BLOCK_WIDTH>>>(d_P, d_N, d_M);

    // copy the result out
    cudaMemcpy(h_P, d_P, WIDTH * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_P);
    cudaFree(d_N);
    cudaFree(d_M);
}

void check_res(float *h_P, float *h_N, float *h_M)
{
    for (int index_o = 0; index_o < WIDTH; index_o++)
    {
        int index_i = index_o - (MASK_WIDTH - 1) / 2;
        float ref = 0.0;
        for (int j = 0; j < MASK_WIDTH; j++)
            if (index_i + j >= 0 && index_i + j < WIDTH)
                ref += h_N[index_i + j] * h_M[j];
        if (abs(ref - h_P[index_o]) >= 0.001)
        {
            printf("INCORRECT!\n");
            printf("h_P[%d] = %f\n", index_o, h_P[index_o]);
            printf("ref = %f\n", ref);
            exit(0);
        }
    }
    printf("CORRECT!\n");
}

int main()
{
    srand(time(0));
    init();                 // initialize the input data and the mask
    cuda_conv_1d(P, N, M);  // calculate in parallel
    check_res(P, N, M);     // check the result of the CUDA program with a sequential program
    return 0;
}