#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#define VECTOR_SIZE 100000000

int a[VECTOR_SIZE];
int b[VECTOR_SIZE];
int c[VECTOR_SIZE];

void init_vec(int *a, int *b, int size);
void vec_add(int *c, int *a, int *b, int size);

int main()
{
    srand(time(NULL));
    init_vec(a, b, VECTOR_SIZE);

    // prepare timer
    struct timeval begin, end;
    double elapsed_sec;

    printf("Serial code starts ...\n");
    gettimeofday(&begin, NULL);
    vec_add(c, a, b, VECTOR_SIZE);
    gettimeofday(&end, NULL);
    elapsed_sec = (double)(end.tv_sec - begin.tv_sec) + (double)(end.tv_usec - begin.tv_usec) / 1000000.0;
    printf("Serial code time: %lf sec\n", elapsed_sec);

    return 0;
}

void init_vec(int *a, int *b, int size)
{
    for (int i = 0; i < size; i++)
    {
        a[i] = rand();
        b[i] = rand();
    }
}

void vec_add(int *c, int *a, int *b, int size)
{
    for (int i = 0; i < size; i++)
    {
        c[i] = a[i] + b[i];
    }
}