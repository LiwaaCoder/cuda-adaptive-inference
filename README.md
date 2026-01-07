# Adaptive Precision Inference Engine (C++/CUDA)

A low-level inference runtime that dynamically switches between FP32 (Standard) and FP16 (Half-Precision) kernels based on task criticality.

## Key Features

*   **Custom CUDA Kernels**: Implemented Shared Memory Tiling and Memory Coalescing manually (No cuBLAS).
*   **Dynamic Dispatch**: Runtime logic to route "Creative" tasks to Tensor Cores (FP16) and "Critical" tasks to CUDA Cores (FP32).
*   **Performance**: Achieved ~12% latency reduction on Nvidia T4 hardware via half-precision optimization.

**Tech Stack**: C++17, CUDA 11.x, Nvidia T4 GPU.
