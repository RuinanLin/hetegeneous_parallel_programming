#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>

#define MESSAGE_SIZE 100000000

#define LOW (-10.0)
#define HIGH 10.0

# define CUDA_SAFE_CALL(call) {                                                 \
    cudaError err = call;                                                       \
    if (cudaSuccess != err) {                                                   \
        fprintf(stderr, "error %d: Cuda error in file '%s' in line %i : %s.\n", \
                err, __FILE__, __LINE__, cudaGetErrorString(err));              \
        exit(EXIT_FAILURE);                                                     \
    }                                                                           \
}

void init_message(float *message, int size);

int main()
{
    // initialize message
    float *message;
    CUDA_SAFE_CALL(cudaHostAlloc((void **)&message, MESSAGE_SIZE * sizeof(float), cudaHostAllocDefault));
    init_message(message, MESSAGE_SIZE);

    // get the device_count
    int device_count;
    CUDA_SAFE_CALL(cudaGetDeviceCount(&device_count));

    // check cudaDevAttrUnifiedAddressing for each device
    for (int device_idx = 0; device_idx < device_count; device_idx++)
    {
        int unifiedAddr_flag;
        CUDA_SAFE_CALL(cudaDeviceGetAttribute(&unifiedAddr_flag, cudaDevAttrUnifiedAddressing, device_idx));
        printf("Device %d's cudaDevAttrUnifiedAddressing is %d\n", device_idx, unifiedAddr_flag);
        if (unifiedAddr_flag == 0)
        {
            CUDA_SAFE_CALL(cudaFreeHost(message));
            printf("System cannot support.\n");
            return 0;
        }
    }

    // alloc device memory on each device
    float **d_message = (float **)malloc(device_count * sizeof(float *));
    for (int device_idx = 0; device_idx < device_count; device_idx++)
    {
        CUDA_SAFE_CALL(cudaSetDevice(device_idx));
        CUDA_SAFE_CALL(cudaMalloc((void **)&(d_message[device_idx]), MESSAGE_SIZE * sizeof(float)));
    }

    // initialize the memory in device 0
    CUDA_SAFE_CALL(cudaSetDevice(0));
    CUDA_SAFE_CALL(cudaMemcpy(d_message[0], message, MESSAGE_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    // copy
    for (int device_idx = 0; device_idx < device_count - 1; device_idx++)
    {
        printf("Device %d passing data to device %d ...\n", device_idx, device_idx + 1);
        CUDA_SAFE_CALL(cudaMemcpy(d_message[device_idx + 1], d_message[device_idx], MESSAGE_SIZE * sizeof(float), cudaMemcpyDefault));
    }

    // copy from the last device to host
    float *res_message = (float *)malloc(MESSAGE_SIZE * sizeof(float));
    CUDA_SAFE_CALL(cudaSetDevice(device_count - 1));
    CUDA_SAFE_CALL(cudaMemcpy(res_message, d_message[device_count - 1], MESSAGE_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    // check whether the message is nearly same
    int i;
    for (i = 0; i < MESSAGE_SIZE; i++)
    {
        if (abs(res_message[i] - message[i]) >= 0.01)
        {
            printf("INCORRECT!\nmessage[%d] = %f\nres_message[%d] = %f\n", i, message[i], i, res_message[i]);
            break;
        }
    }
    if (i == MESSAGE_SIZE)
        printf("CORRECT!\n");

    // free the memory
    CUDA_SAFE_CALL(cudaFreeHost(message));
    for (int device_idx = 0; device_idx < device_count; device_idx++)
    {
        CUDA_SAFE_CALL(cudaSetDevice(device_idx));
        CUDA_SAFE_CALL(cudaFree(d_message[device_idx]));
    }
    free(d_message);
    free(res_message);
    return 0;
}

float f_rand(float low, float high)
{
    return (float)rand() / (float)RAND_MAX * (high - low) + low;
}

void init_message(float *message, int size)
{
    srand(time(NULL));
    for (int i = 0; i < size; i++)
        message[i] = f_rand(LOW, HIGH);
}