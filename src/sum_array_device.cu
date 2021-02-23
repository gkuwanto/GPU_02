#include <cassert>
#include <cuda_runtime.h>
#include "sum_array_device.cuh"

__global__
void naiveSumArray(const float *input, float *output, int n) {
    double sum = 0.0;
    //reduce multiple elements per thread
    for (int i=1;i<32;i++)
        int index = blockIdx.x * blockDim.x + threadIdx.x + gridDim.x * blockDim.x * i;
        if (index < n)
            sum += input[index];
    atomicAdd(output, sum);
}


void cudaSumArray(
    const float *d_input,
    float *d_output,
    int n,
    SumArrayImplementation type)
{
    if (type == NAIVE) {
        dim3 blockSize(1024, 1);
        dim3 gridSize(n / 32 / 1024, 1);
        naiveSumArray<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
}