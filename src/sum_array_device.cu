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
    for(unsigned int s = 1; s< blockDim.x; s*= 2) {
        if (tid % (2*s) == 0) {
            sdata[tid] += sdata[tid+s];
        }
        __syncthreads();
    }
    if (tid == 0) atomicAdd(output, sdata[0]);
}

__global__
void nonDivergentSumArray(const int *input, int *output, int n) {
    __shared__ int sdata[1024];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = input[i];
    __syncthreads();
    for(unsigned int s = 1; s< blockDim.x; s*= 2) {
        int index = 2 * s * tid;
        if (index < blockDim.x) {
            sdata[index] += sdata[index+s];
        }
        __syncthreads();
    }
    if (tid == 0) atomicAdd(output, sdata[0]);
}

__global__
void sequentialSumArray(const int *input, int *output, int n) {
    __shared__ int sdata[1024];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = input[i];
    __syncthreads();
    for(unsigned int s blockDim.x / 2; s>0; s >>= 1) {
        if(tid < s) {
            sdata[tid] += sdata[tid + s]
        }
        __syncthreads();
    }
    if (tid == 0) atomicAdd(output, sdata[0]);
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
    if (type == BINARY) {
        dim3 blockSize(1024, 1);
        dim3 gridSize(n/1024, 1);
        binarySumArray<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    if (type == NONDIV) {
        dim3 blockSize(1024, 1);
        dim3 gridSize(n/1024, 1);
        nonDivergentSumArray<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    if (type == SEQUENTIAL) {
        dim3 blockSize(1024, 1);
        dim3 gridSize(n/1024, 1);
        sequentialSumArray<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    
}