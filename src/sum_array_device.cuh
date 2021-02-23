#ifndef CUDA_SUM_CUH
#define CUDA_SUM_CUH

enum SumArrayImplementation { NAIVE, BINARY, NONDIV, SEQUENTIAL };

void cudaSumArray(
    const float *d_input,
    float *d_output,
    int n,
    SumArrayImplementation type);

#endif
