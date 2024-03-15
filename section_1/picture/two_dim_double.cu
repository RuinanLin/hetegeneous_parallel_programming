#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>

#define CHANNEL 3
#define HEIGHT 100
#define WIDTH 100

#define VALUE_RANGE 128.0

uint8_t pic_in[CHANNEL * HEIGHT * WIDTH];
uint8_t pic_out[CHANNEL * HEIGHT * WIDTH];

__global__
void pic_double_kernel(uint8_t *d_pic_out, uint8_t *d_pic_in, int channel, int height, int width)
{
    int c = threadIdx.x;
    int h = threadIdx.y + blockDim.y * blockIdx.y;
    int w = threadIdx.z + blockDim.z * blockIdx.z;
    if (h < height && w < width)
    {
        d_pic_out[c*height*width + h*width + w] = d_pic_in[c*height*width + h*width + w] << 1;
    }
}

void pic_double(uint8_t *pic_out, uint8_t *pic_in, int channel, int height, int width)
{
    int size = channel * height * width * sizeof(uint8_t);
    uint8_t *d_pic_out;
    uint8_t *d_pic_in;
    cudaMalloc((void **)&d_pic_in, size);
    cudaMemcpy(d_pic_in, pic_in, size, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_pic_out, size);

    dim3 DimGrid(1, (height-1)/16+1, (width-1)/16+1);
    dim3 DimBlock(3, 16, 16);
    pic_double_kernel<<<DimGrid, DimBlock>>>(d_pic_out, d_pic_in, channel, height, width);

    cudaMemcpy(pic_out, d_pic_out, size, cudaMemcpyDeviceToHost);
    cudaFree(d_pic_out);
    cudaFree(d_pic_in);
}

int main()
{
    srand(time(0));
    for (int c = 0; c < CHANNEL; c++)
        for (int h = 0; h < HEIGHT; h++)
            for (int w = 0; w < WIDTH; w++)
                pic_in[c*HEIGHT*WIDTH + h*WIDTH + w] = (uint8_t)((float)rand() / (float)RAND_MAX * VALUE_RANGE);

    pic_double(pic_out, pic_in, CHANNEL, HEIGHT, WIDTH);
    printf("pic_in[1][45][12] = %d\n", pic_in[1*HEIGHT*WIDTH + 45*WIDTH + 12]);
    printf("pic_out[1][45][12] = %d\n", pic_out[1*HEIGHT*WIDTH + 45*WIDTH + 12]);

    return 0;
}