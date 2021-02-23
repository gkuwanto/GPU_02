#include <cassert>
#include <cuda_runtime.h>
#include "sum_array_device.cuh"

__global__
void naiveSumArray(const float *input, float *output, int n) {
    float partial_sum = 0.0;
    //reduce multiple elements per thread
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
            i < n; 
            i += blockDim.x * gridDim.x) {
        partial_sum += input[i];
    }
    atomicAdd(output, partial_sum);
}

__global__
void sequentialSumArray(const float *input, float *output, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ float sdata[1024];
    if (idx < n){

        /*copy to shared memory*/
        sdata[threadIdx.x] = input[idx];
        __syncthreads();

        for(int stride=blockDim.x/2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride) {
                float lhs = sdata[threadIdx.x];
                float rhs = sdata[threadIdx.x + stride];
                sdata[threadIdx.x] = lhs+rhs;
            }
            __syncthreads();
        }
    }
    if (idx == 0) output = sdata[0];
}


void cudaSumArray(
    const float *d_input,
    float *d_output,
    int n,
    SumArrayImplementation type)
{
    if (type == NAIVE) {
        dim3 blockSize(1024, 1);
        dim3 gridSize(n / 1024, 1);
        naiveSumArray<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    if (type == SEQUENTIAL) {
        dim3 blockSize(1024, 1);
        dim3 gridSize(n / 1024 + 1, 1);
        naiveSumArray<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
}