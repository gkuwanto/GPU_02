#include <cassert>
#include <cuda_runtime.h>
#include "sum_array_device.cuh"

__global__
void naiveSumArray(const float *input, float *output, int n) {
    int index = blockIdx.x * blockDim.x * 32 + threadIdx.x;
    double sum = 0;
    for (int i = 0; i < 32; i++){
        sum+=input[index+i*blockDim.x];
    }
    atomicAdd(output, sum);
}


void cudaSumArray(
    const float *d_input,
    float *d_output,
    int n,
    SumArrayImplementation type)
{
    if (type == NAIVE) {
        dim3 blockSize(32, 1);
        dim3 gridSize(n / 32, 1);
        naiveSumArray<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
}