#include <cuda.h>
#include <stdio.h>
#include <time.h>

#define M 1000
#define N 2000
#define K 1500

#define LOW (-10.0)
#define HIGH 10.0

#define TILE_WIDTH 16

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
    __shared__ double ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ double ds_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = bx * blockDim.x + tx;
    int Col = by * blockDim.y + ty;
    double Cvalue = 0.0;

    // load the elements
    for (int t = 0; t < (K - 1) / TILE_WIDTH + 1; t++)
    {
        // load A
        if (Row < M && t * TILE_WIDTH + ty < K)
            ds_A[tx][ty] = d_a[Row * K + t * TILE_WIDTH + ty];
        else
            ds_A[tx][ty] = 0.0;

        // load B
        if (t * TILE_WIDTH + tx < K && Col < N)
            ds_B[tx][ty] = d_b[(t * TILE_WIDTH + tx) * N + Col];
        else
            ds_B[tx][ty] = 0.0;

        // sync
        __syncthreads();

        // update partial sum
        for (int i = 0; i < TILE_WIDTH; i++)
            Cvalue += ds_A[tx][i] * ds_B[i][ty];

        // sync
        __syncthreads();
    }

    // store the result
    if (Row < M && Col < N)
        d_c[Row * N + Col] = Cvalue;
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
    
    dim3 dim_grid((M-1)/TILE_WIDTH+1, (N-1)/TILE_WIDTH+1, 1);
    dim3 dim_block(TILE_WIDTH, TILE_WIDTH, 1);
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