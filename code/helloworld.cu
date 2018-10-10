#include <cuda_runtime.h>
#include <stdio.h>

__global__ void HelloWorldFromGPU(void)
{
    printf("Hello world from GPU!\n");
}

int main()
{
    printf("Hello world from CPU!\n");
    HelloWorldFromGPU<<<1,10>>>();
    cudaDeviceReset();
    return 0;
}
