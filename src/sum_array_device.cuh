#ifndef CUDA_SUM_CUH
#define CUDA_SUM_CUH

enum SumArrayImplementation { NAIVE, BINARY, NONDIV, SEQUENTIAL };

void cudaSumArray(
    const int *d_input,
    int *d_output,
    int n,
    SumArrayImplementation type);

#endif
