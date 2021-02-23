#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <cuda_runtime.h>

#include "sum_array_device.cuh"

#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
    bool abort = true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n",
            cudaGetErrorString(code), file, line);
        exit(code);
    }
}

void cpuSumArray(int* input, int& output, int n) {
    int sum = 0;
    for (int i = 0; i<n; ++i){
        sum += input[i];
    }
    output = sum;
}

void checkSumArray(int* input, int result, int n) {
    int output;
    cpuSumArray(input, output, n);

    if (output!=result)
        fprintf(stderr, "Sum is different: CPU %d != %d GPU\n",output, result);
        assert(output==result);
}

void randomFill(int *fill, int size) {
    for (int i = 0; i < size; i++) {
        int r = (rand() % 100) - 50; //random range from -50 to 50
        fill[i] = r;
    }
}

int main(int argc, char *argv[]) {
    //seed RNG
    srand(2016);

    //default parameters
    std::string kernel = "all";
    //pow 2
    int size_to_run = -1;


    assert(argc <= 2);
    if (argc >= 2)
        size_to_run = atoi(argv[1]);
    if (argc == 3)
        kernel = argv[2];

    if (size_to_run>20 || size_to_run<10) {
        fprintf(stderr,
            "Program only designed to run sizes 2^10 to 2^20\n");
    }
    
    assert(kernel == "all"        ||
        kernel == "cpu"           ||
        kernel == "naive"         ||
        kernel == "binary"        ||
        kernel == "non_divergent" ||
        kernel == "sequential");

    for (int _i = 10; _i < 21; _i++) {
        if (size_to_run != -1 && size_to_run != _i)
            continue;

        int n = 1 << _i;

        cudaEvent_t start;
        cudaEvent_t stop;

#define START_TIMER() {                                                        \
            gpuErrChk(cudaEventCreate(&start));                                \
            gpuErrChk(cudaEventCreate(&stop));                                 \
            gpuErrChk(cudaEventRecord(start));                                 \
        }

#define STOP_RECORD_TIMER(name) {                                              \
            gpuErrChk(cudaEventRecord(stop));                                  \
            gpuErrChk(cudaEventSynchronize(stop));                             \
            gpuErrChk(cudaEventElapsedTime(&name, start, stop));               \
            gpuErrChk(cudaEventDestroy(start));                                \
            gpuErrChk(cudaEventDestroy(stop));                                 \
        }

        float cpu_ms = -1;
        float naive_gpu_ms = -1;
        float binary_gpu_ms = -1;
        float non_divergent_gpu_ms = -1;
        float sequential_gpu_ms = -1;

        int *input = new int[n];
        int output = 0;

        int *d_input;
        int *d_output;
        gpuErrChk(cudaMalloc(&d_input, n * sizeof(int)));
        gpuErrChk(cudaMalloc(&d_output, sizeof(int)));

        randomFill(input, n);


        gpuErrChk(cudaMemcpy(d_input, input, n * sizeof(int), 
            cudaMemcpyHostToDevice));

        if (kernel == "cpu" || kernel == "all") {
            START_TIMER();
            cpuSumArray(input, output, n);
            STOP_RECORD_TIMER(cpu_ms);
            printf("Size %d CPU only: %f ms\n", n, cpu_ms);
        }


        if (kernel == "naive" || kernel == "all") {
            START_TIMER();
            cudaSumArray(d_input, d_output, n, NAIVE);
            STOP_RECORD_TIMER(naive_gpu_ms);

            gpuErrChk(cudaMemcpy(&output, d_output, sizeof(int), 
                cudaMemcpyDeviceToHost));
            checkSumArray(input, output, n);
            output = 0;
            gpuErrChk(cudaMemset(d_output, 0, sizeof(int)));

            printf("Size %d naive GPU: %f ms\n", n, naive_gpu_ms);
        }

        if (kernel == "binary" || kernel == "all") {
            START_TIMER();
            cudaSumArray(d_input, d_output, n, binary);
            STOP_RECORD_TIMER(binary_gpu_ms);

            gpuErrChk(cudaMemcpy(&output, d_output, sizeof(int), 
                cudaMemcpyDeviceToHost));
            checkSumArray(input, output, n);
            output = 0;
            gpuErrChk(cudaMemset(d_output, 0, sizeof(int)));

            printf("Size %d binary tree GPU: %f ms\n", n, binary_gpu_ms);
        }
    }
    
    
}
