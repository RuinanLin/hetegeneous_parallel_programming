#include <stdio.h>
#include <omp.h>

int main()
{
    omp_set_num_threads(4);
    #pragma omp parallel for
    for (int device_idx = 0; device_idx < 4; device_idx++)
        printf("%d on %d\n", device_idx, omp_get_thread_num());
    return 0;
}