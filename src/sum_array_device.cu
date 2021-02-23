#include <cassert>
#include <cuda_runtime.h>
#include "sum_array_device.cuh"

__global__
void naiveSumArray(const float *input, float *output, int n) {
    double sum = 0.0;
    //reduce multiple elements per thread
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
            i < n; 
            i += blockDim.x * gridDim.x) {
        sum += input[i];
    }
    atomicAdd(output, sum);
}


void cudaSumArray(
    const float *d_input,
    float *d_output,
    int n,
    SumArrayImplementation type);
{
    if (type == NAIVE) {
        dim3 blockSize(1024, 1);
        dim3 gridSize(n / 1024, 1);
        naiveSumArray<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
}