// gpu_common.cuh
// Small helpers shared by every kernel file. Only included from .cu files.

#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// Abort with a message if a CUDA call fails.
#define cudaCheck(call)                                                     \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                \
                         cudaGetErrorString(err), __FILE__, __LINE__);      \
            std::exit(1);                                                   \
        }                                                                   \
    } while (0)

// One thread per pixel, 16x16 threads per block.
static const int kBlockDim = 16;

inline dim3 blockFor() {
    return dim3(kBlockDim, kBlockDim);
}

inline dim3 gridFor(int n) {
    int blocks = (n + kBlockDim - 1) / kBlockDim;
    return dim3(blocks, blocks);
}

// Clamp a coordinate into [0, n-1]. This is BORDER_REPLICATE, matching the
// OpenCV calls on the CPU side.
__device__ __forceinline__ int clampIdx(int i, int n) {
    return i < 0 ? 0 : (i >= n ? n - 1 : i);
}

// Read pixel (x, y) with replicated borders.
__device__ __forceinline__ unsigned char pixelAt(const unsigned char* img,
                                                 int x, int y, int n) {
    return img[clampIdx(y, n) * n + clampIdx(x, n)];
}

// Round a float to the nearest integer and clamp to [0, 255].
__device__ __forceinline__ unsigned char toByte(float v) {
    int r = __float2int_rn(v);
    return (unsigned char)(r < 0 ? 0 : (r > 255 ? 255 : r));
}
