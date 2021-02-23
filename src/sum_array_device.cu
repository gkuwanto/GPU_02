#include <cassert>
#include <cuda_runtime.h>
#include "sum_array_device.cuh"

__global__
void naiveSumArray(const int *input, int *output, int n) {
    int partial_sum = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
    {
        int val = input[i];
        partial_sum += val;
    }
    atomicAdd(output, partial_sum);
}

__global__
void binarySumArray(const int *input, int *output, int n) {
    __shared__ int sdata[1024];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = input[i];
    __syncthreads();
    for(unsigned int s = 1; s< blockDim.x; x*= 2) {
        if (tid % (2*s) == 0) {
            sdata[tid] += sdata[tid+s];
        }
        __syncthreads();
    }
}


void cudaSumArray(
    const int *d_input,
    int *d_output,
    int n,
    SumArrayImplementation type)
{
    if (type == NAIVE) {
        dim3 blockSize(1024, 1);
        dim3 gridSize(1, 1);
        naiveSumArray<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
}