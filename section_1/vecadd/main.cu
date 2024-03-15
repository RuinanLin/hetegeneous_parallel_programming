#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define VECTOR_SIZE 10000
#define VALUE_RANGE 100

void checkError(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
}

__global__
void vecAddKernel(float *A, float *B, float *C, int n)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n)
        C[i] = A[i] + B[i];
}

void vecAdd(float *h_A, float *h_B, float *h_C, int n)
{
    int size = n * sizeof(float);
    float *d_A;
    float *d_B;
    float *d_C;

    cudaError_t err_A = cudaMalloc((void **)&d_A, size); checkError(err_A);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaError_t err_B = cudaMalloc((void **)&d_B, size); checkError(err_B);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    cudaError_t err_C = cudaMalloc((void **)&d_C, size); checkError(err_C);

    vecAddKernel<<<(n - 1) / 256 + 1, 256>>>(d_A, d_B, d_C, n);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main()
{
    float *h_A = (float *)malloc(VECTOR_SIZE * sizeof(float));
    float *h_B = (float *)malloc(VECTOR_SIZE * sizeof(float));
    float *h_C = (float *)malloc(VECTOR_SIZE * sizeof(float));
    srand(time(0));
    for (int i = 0; i < VECTOR_SIZE; i++)
    {
        h_A[i] = (float)rand() / ((float)RAND_MAX / VALUE_RANGE);
        h_B[i] = (float)rand() / ((float)RAND_MAX / VALUE_RANGE);
    }
    vecAdd(h_A, h_B, h_C, VECTOR_SIZE);
    for (int i = 0; i < VECTOR_SIZE; i++)
        printf("%f + %f = %f\n", h_A[i], h_B[i], h_C[i]);
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}
