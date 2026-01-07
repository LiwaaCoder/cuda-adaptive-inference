#ifndef ENGINE_H
#define ENGINE_H

#include <string>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

class AdaptiveComputeEngine {
public:
    int N;
    float *h_A, *h_B, *h_C; // Host Memory
    float *d_A_32, *d_B_32, *d_C_32; // Device Memory (FP32)
    __half *d_A_16, *d_B_16, *d_C_16; // Device Memory (FP16)
    
    AdaptiveComputeEngine(int size);
    ~AdaptiveComputeEngine();
    
    void runInference(std::string taskType);
};

#endif // ENGINE_H
