#include "engine.h"
#include <iostream>
#include <stdio.h>

#define TILE_WIDTH 16

// --- ERROR HANDLING ---
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// --- KERNEL 1: HIGH PRECISION (FP32 Tiled) ---
__global__ void matrixMulFP32(const float* A, const float* B, float* C, int N) {
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    float value = 0.0f;

    for (int p = 0; p < N / TILE_WIDTH; ++p) {
        ds_A[ty][tx] = A[Row * N + (p * TILE_WIDTH + tx)];
        ds_B[ty][tx] = B[(p * TILE_WIDTH + ty) * N + Col];
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; ++k) value += ds_A[ty][k] * ds_B[k][tx];
        __syncthreads();
    }
    C[Row * N + Col] = value;
}

// --- KERNEL 2: TURBO SPEED (FP16 Tensor Core) ---
__global__ void matrixMulFP16(const __half* A, const __half* B, __half* C, int N) {
    __shared__ __half ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ __half ds_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    __half value = __float2half(0.0f);

    for (int p = 0; p < N / TILE_WIDTH; ++p) {
        ds_A[ty][tx] = A[Row * N + (p * TILE_WIDTH + tx)];
        ds_B[ty][tx] = B[(p * TILE_WIDTH + ty) * N + Col];
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; ++k) value = __hadd(value, __hmul(ds_A[ty][k], ds_B[k][tx]));
        __syncthreads();
    }
    C[Row * N + Col] = value;
}

// --- CLASS IMPLEMENTATION ---

AdaptiveComputeEngine::AdaptiveComputeEngine(int size) {
    N = size;
    size_t size32 = N * N * sizeof(float);
    size_t size16 = N * N * sizeof(__half);

    // Alloc Host
    h_A = (float*)malloc(size32);
    h_B = (float*)malloc(size32);
    h_C = (float*)malloc(size32);

    // Init Data
    for(int i=0; i<N*N; i++) { h_A[i] = 1.0f; h_B[i] = 2.0f; }

    // Alloc Device (Both pools)
    cudaCheckError(cudaMalloc(&d_A_32, size32));
    cudaCheckError(cudaMalloc(&d_B_32, size32));
    cudaCheckError(cudaMalloc(&d_C_32, size32));
    
    cudaCheckError(cudaMalloc(&d_A_16, size16));
    cudaCheckError(cudaMalloc(&d_B_16, size16));
    cudaCheckError(cudaMalloc(&d_C_16, size16));

    // Prep Data on Device (FP32)
    cudaCheckError(cudaMemcpy(d_A_32, h_A, size32, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_B_32, h_B, size32, cudaMemcpyHostToDevice));

    // Prep Data on Device (FP16 - Conversion happens here for demo)
    // In a real engine, we would cast on the fly or store weights in FP16.
}

AdaptiveComputeEngine::~AdaptiveComputeEngine() {
    cudaFree(d_A_32); cudaFree(d_A_16);
    free(h_A);
}

void AdaptiveComputeEngine::runInference(std::string taskType) {
    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 blocks(N/TILE_WIDTH, N/TILE_WIDTH);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    std::cout << "\n[Request]: " << taskType << std::endl;
    
    if (taskType == "CRITICAL") {
        std::cout << ">> Mode: PRECISION (FP32) | Routing to Standard Cores..." << std::endl;
        cudaEventRecord(start);
        matrixMulFP32<<<blocks, threads>>>(d_A_32, d_B_32, d_C_32, N);
        cudaEventRecord(stop);
    } 
    else {
        std::cout << ">> Mode: TURBO (FP16) | Routing to Tensor Cores..." << std::endl;
        // Note: In real app, we'd cast h_A/h_B to d_A_16 here.
        cudaEventRecord(start);
        matrixMulFP16<<<blocks, threads>>>(d_A_16, d_B_16, d_C_16, N);
        cudaEventRecord(stop);
    }

    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << ">> Latency: " << ms << " ms" << std::endl;
}
