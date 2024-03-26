#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#define NUM_THREADS 24

#define M 2000
#define K 3000
#define N 1000

#define LOW (-10.0)
#define HIGH 10.0

double A[M * K];
double B[K * N];
double C[M * N];
double C_ref[M * N];

double d_rand(double low, double high);
void init_mat(double *mat, int size);
void omp_matrix_mult(double *c, double *a, double *b, int tile_size, int thread_num);
void serial_matrix_mult(double *c, double *a, double *b);
void check_res(double *c, double *c_ref, int size);

int main()
{
    // initialize input matrices
    srand(time(NULL));
    init_mat(A, M * K);
    init_mat(B, K * N);

    // set the number of threads
    omp_set_num_threads(NUM_THREADS);

    // set the tile size
    int tile_size = (M - 1) / NUM_THREADS + 1;

    // prepare timer
    struct timeval begin, end;
    double elapsed_sec;

    // openmp matrix multiplication
    printf("Parallel code starts ...\n");
    gettimeofday(&begin, NULL);
    #pragma omp parallel
        omp_matrix_mult(C, A, B, tile_size, omp_get_thread_num());
    gettimeofday(&end, NULL);
    elapsed_sec = (double)(end.tv_sec - begin.tv_sec) + (double)(end.tv_usec - begin.tv_usec) / 1000000.0;
    printf("Parallel code time: %lf sec\n", elapsed_sec);

    // serial code
    printf("Serial code starts ...\n");
    gettimeofday(&begin, NULL);
    serial_matrix_mult(C_ref, A, B);
    gettimeofday(&end, NULL);
    elapsed_sec = (double)(end.tv_sec - begin.tv_sec) + (double)(end.tv_usec - begin.tv_usec) / 1000000.0;
    printf("Serial code time: %lf sec\n", elapsed_sec);

    // check the result
    check_res(C, C_ref, M * N);
    return 0;
}

double d_rand(double low, double high)
{
    return (double)rand() / (double)RAND_MAX * (high - low) + low;
}

void init_mat(double *mat, int size)
{
    for (int i = 0; i < size; i++)
        mat[i] = d_rand(LOW, HIGH);
}

void omp_matrix_mult(double *c, double *a, double *b, int tile_size, int thread_num)
{
    int rid = thread_num * tile_size;
    for (int row = rid; row < ((M < rid + tile_size) ? M : rid + tile_size); row++)
    {
        for (int col = 0; col < N; col++)
        {
            double sum = 0;
            for (int k = 0; k < K; k++)
                sum += a[row * K + k] * b[k * N + col];
            c[row * N + col] = sum;
        }
    }
}

void serial_matrix_mult(double *c, double *a, double *b)
{
    for (int row = 0; row < M; row++)
    {
        for (int col = 0; col < N; col++)
        {
            double sum = 0;
            for (int k = 0; k < K; k++)
                sum += a[row * K + k] * b[k * N + col];
            c[row * N + col] = sum;
        }
    }
}

void check_res(double *c, double *c_ref, int size)
{
    for (int i = 0; i < size; i++)
    {
        if (abs(c[i] - c_ref[i]) >= 0.01)
        {
            printf("INCORRECT!\n");
            printf("c[%d] = %lf\n", i, c[i]);
            printf("c_ref[%d] = %lf\n", i, c_ref[i]);
            return;
        }
    }
    printf("CORRECT!\n");
    return;
}