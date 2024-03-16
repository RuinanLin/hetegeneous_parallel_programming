#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define CHANNEL 3
#define IMAGE_SIZE 1000
#define MASK_SIZE 5

#define BLOCK_SIZE 32
#define O_TILE_WIDTH (BLOCK_SIZE - MASK_SIZE + 1)

#define LOW (-10.0)
#define HIGH 10.0

float P[IMAGE_SIZE * IMAGE_SIZE];
float N[IMAGE_SIZE * IMAGE_SIZE * CHANNEL];
float M[MASK_SIZE * MASK_SIZE * CHANNEL];


float d_rand(float low, float high)
{
    return (float)rand() / (float)RAND_MAX * (high - low) + low;
}

void init_input_values()
{
    for (int i = 0; i < IMAGE_SIZE * IMAGE_SIZE * CHANNEL; i++)
        N[i] = d_rand(LOW, HIGH);
}

void init_mask_values()
{
    for (int i = 0; i < MASK_SIZE * MASK_SIZE * CHANNEL; i++)
        M[i] = d_rand(LOW, HIGH);
}

__global__
void cuda_conv_2d_kernel(float *d_P, float *d_N, const float * __restrict__ d_M)
{
    // who am i
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int index_o_row = blockIdx.y * O_TILE_WIDTH + ty;
    int index_o_col = blockIdx.x * O_TILE_WIDTH + tx;
    int index_i_row = index_o_row - (MASK_SIZE - 1) / 2;
    int index_i_col = index_o_col - (MASK_SIZE - 1) / 2;

    // load the tile to the shared memory
    __shared__ float Ns[CHANNEL][BLOCK_SIZE][BLOCK_SIZE];
    if (index_i_row >= 0 && index_i_row < IMAGE_SIZE && index_i_col >= 0 && index_i_col < IMAGE_SIZE)
        for (int c = 0; c < CHANNEL; c++)
            Ns[c][ty][tx] = d_N[c * IMAGE_SIZE * IMAGE_SIZE + index_i_row * IMAGE_SIZE + index_i_col];
    else
        for (int c = 0; c < CHANNEL; c++)
            Ns[c][ty][tx] = 0.0f;
    __syncthreads();

    // calculate conv
    if (ty < O_TILE_WIDTH && tx < O_TILE_WIDTH)
    {
        float res = 0.0f;
        for (int c = 0; c < CHANNEL; c++)
            for (int i = 0; i < MASK_SIZE; i++)
                for (int j = 0; j < MASK_SIZE; j++)
                    res += Ns[c][ty + i][tx + j] * d_M[c * MASK_SIZE * MASK_SIZE + i * MASK_SIZE + j];
        if (index_o_row < IMAGE_SIZE && index_o_col < IMAGE_SIZE)
            d_P[index_o_row * IMAGE_SIZE + index_o_col] = res;
    }
}

void cuda_conv_2d(float *h_P, float *h_N, float *h_M)
{
    // alloc space on device GPU
    float *d_P;
    float *d_N;
    float *d_M;
    cudaMalloc((void **)&d_P, IMAGE_SIZE * IMAGE_SIZE * sizeof(float));
    cudaMalloc((void **)&d_N, IMAGE_SIZE * IMAGE_SIZE * CHANNEL * sizeof(float));
    cudaMemcpy(d_N, h_N, IMAGE_SIZE * IMAGE_SIZE * CHANNEL * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_M, MASK_SIZE * MASK_SIZE * CHANNEL * sizeof(float));
    cudaMemcpy(d_M, h_M, MASK_SIZE * MASK_SIZE * CHANNEL * sizeof(float), cudaMemcpyHostToDevice);

    // launch the kernel function
    dim3 DimGrid((IMAGE_SIZE - 1) / O_TILE_WIDTH + 1, (IMAGE_SIZE - 1) / O_TILE_WIDTH + 1, 1);
    dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    cuda_conv_2d_kernel<<<DimGrid, DimBlock>>>(d_P, d_N, d_M);

    // collect the result
    cudaMemcpy(h_P, d_P, IMAGE_SIZE * IMAGE_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // free the memory in the device
    cudaFree(d_P);
    cudaFree(d_N);
    cudaFree(d_M);
}

void check_res(float *h_P, float *h_N, float *h_M)
{
    for (int p_row = 0; p_row < IMAGE_SIZE; p_row++)
        for (int p_col = 0; p_col < IMAGE_SIZE; p_col++)
        {
            float ref = 0.0f;
            int index_i_row = p_row - (MASK_SIZE - 1) / 2;
            int index_i_col = p_col - (MASK_SIZE - 1) / 2;
            for (int c = 0; c < CHANNEL; c++)
                for (int m_row = 0; m_row < MASK_SIZE; m_row++)
                    for (int m_col = 0; m_col < MASK_SIZE; m_col++)
                        if (index_i_row + m_row >= 0 && index_i_row + m_row < IMAGE_SIZE && index_i_col + m_col >= 0 && index_i_col + m_col < IMAGE_SIZE)
                            ref += h_N[c * IMAGE_SIZE * IMAGE_SIZE + (index_i_row + m_row) * IMAGE_SIZE + (index_i_col + m_col)] * h_M[c * MASK_SIZE * MASK_SIZE + m_row * MASK_SIZE + m_col];
            if (abs(ref - h_P[p_row * IMAGE_SIZE + p_col]) >= 0.001)
            {
                printf("INCORRECT!\n");
                printf("h_P[%d] = %f\n", p_row, h_P[p_row]);
                printf("ref = %f\n", ref);
                exit(0);
            }
        }
    printf("CORRECT!\n");
}

int main()
{
    srand(time(0));
    init_input_values();
    init_mask_values();
    cuda_conv_2d(P, N, M);  // carry out conv on GPU
    check_res(P, N, M);     // check res with sequential logic
    return 0;
}
