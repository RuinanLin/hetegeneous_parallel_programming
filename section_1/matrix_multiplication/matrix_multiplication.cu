#include <cuda.h>
#include <stdio.h>
#include <time.h>

#define M 1000
#define N 2000
#define K 1500

#define LOW (-10.0)
#define HIGH 10.0

double A[M*K];
double B[K*N];
double C[M*N];

double d_rand(double low, double high)
{
    return (double)rand() / (double)RAND_MAX * (high - low) + low;
}

void init()
{
    for (int m = 0; m < M; m++)
        for (int k = 0; k < K; k++)
            A[m*K+k] = d_rand(LOW, HIGH);
    for (int k = 0; k < K; k++)
        for (int n = 0; n < N; n++)
            B[k*N+n] = d_rand(LOW, HIGH);
}

__global__
void cuda_matmul_kernel(double *d_c, double *d_a, double *d_b)
{
    int m = threadIdx.x + blockDim.x * blockIdx.x;
    int n = threadIdx.y + blockDim.y * blockIdx.y;
    if (m < M && n < N)
    {
        double partial_sum = 0.0;
        for (int k = 0; k < K; k++)
            partial_sum += d_a[m*K+k] * d_b[k*N+n];
        d_c[m*N+n] = partial_sum;
    }
}

void cuda_matmul(double *C, double *A, double *B)
{
    int size_a = M * K * sizeof(double);
    int size_b = K * N * sizeof(double);
    int size_c = M * N * sizeof(double);
    double *d_a;
    double *d_b;
    double *d_c;
    cudaMalloc((void **)&d_a, size_a);
    cudaMemcpy(d_a, A, size_a, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_b, size_b);
    cudaMemcpy(d_b, B, size_b, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_c, size_c);
    
    dim3 dim_grid((M-1)/16+1, (N-1)/16+1, 1);
    dim3 dim_block(16, 16, 1);
    cuda_matmul_kernel<<<dim_grid, dim_block>>>(d_c, d_a, d_b);

    cudaMemcpy(C, d_c, size_c, cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

void check_res(double *C, double *A, double *B)
{
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
        {
            double partial_sum = 0.0;
            for (int k = 0; k < K; k++)
                partial_sum += A[m*K+k] * B[k*N+n];
            if (partial_sum - C[m*N+n] >= 0.001 || C[m*N+n] - partial_sum >= 0.001)
            {
                printf("INCORRECT\n");
                exit(0);
            }
        }
    printf("CORRECT\n");
}

int main()
{
    srand(time(0));
    init();
    cuda_matmul(C, A, B);
    check_res(C, A, B);
    return 0;
}