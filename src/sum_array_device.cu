#include <cassert>
#include <cuda_runtime.h>
#include "sum_array_device.cuh"

__global__
void naiveSumArray(const float *input, float *output, int n) {
    float partial_sum = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
    {
        float val = input[i];
        partial_sum += val;
    }
    atomicAdd(output, partial_sum);
}

void cudaSumArray(
    const float *d_input,
    float *d_output,
    int n,
    SumArrayImplementation type)
{
    if (type == NAIVE) {
        dim3 blockSize(1024, 1);
        dim3 gridSize(1, 1);
        naiveSumArray<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
}