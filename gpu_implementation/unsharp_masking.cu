// unsharp_masking.cu
// Unsharp masking: dst = src + amount * (src - blur(src)), amount = 1.0,
// which simplifies to dst = 2*src - blur. Two kernels, mirroring the CPU
// side (GaussianBlur then addWeighted). Fusing them into one kernel is an
// obvious optimization to explore.

#include "gpu_filters.h"
#include "gpu_common.cuh"

__global__ void sharpenKernel(const unsigned char* src,
                              const unsigned char* blurred,
                              unsigned char* dst, int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= n || y >= n) return;

    int i = y * n + x;
    dst[i] = toByte(2.0f * src[i] - 1.0f * blurred[i]);
}

void unsharp_gpu(GpuBuffers& g) {
    launchGauss5x5(g.src, g.tmp, g.n);
    sharpenKernel<<<gridFor(g.n), blockFor()>>>(g.src, g.tmp, g.dst, g.n);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
}
