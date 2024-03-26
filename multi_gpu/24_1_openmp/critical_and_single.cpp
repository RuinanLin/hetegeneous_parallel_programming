#include <stdio.h>
#include <omp.h>

int main()
{
    int a = 0;
    int b = 0;
    #pragma omp parallel num_threads(4)
    {
        #pragma omp critical
        a += 1;
        printf("[Thread %d]\n", omp_get_thread_num());
        #pragma omp single
        b += 1;
    }
    printf("a = %d\n", a);
    printf("b = %d\n", b);
    return 0;
}