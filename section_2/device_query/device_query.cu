#include <cuda.h>
#include <stdio.h>

int main()
{
    // number of devices in the system
    int dev_count;
    cudaGetDeviceCount(&dev_count);
    printf("dev_count: %d\n", dev_count);

    // capabilities of devices
    cudaDeviceProp dev_prop;
    for (int i = 0; i < dev_count; i++)
    {
        cudaGetDeviceProperties(&dev_prop, i);
        printf("----------------------------\n");
        printf("maxThreadsPerBlock: %d\n", dev_prop.maxThreadsPerBlock);
        printf("sharedMemoryPerBlock: %d\n", dev_prop.sharedMemPerBlock);
    }

    return 0;
}