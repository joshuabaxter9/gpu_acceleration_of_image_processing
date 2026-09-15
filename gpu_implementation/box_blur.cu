// box_blur.cu
// Normalized box blur: each output pixel is the mean of its (2r+1)x(2r+1)
// neighborhood. One thread per output pixel, reading straight from global
// memory. A shared-memory tiled version is the natural optimization.

#include "gpu_filters.h"
#include "gpu_common.cuh"

__global__ void boxBlurKernel(const unsigned char* src, unsigned char* dst,
                              int n, int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= n || y >= n) return;

    int sum = 0;
    for (int dy = -radius; dy <= radius; ++dy)
        for (int dx = -radius; dx <= radius; ++dx)
            sum += pixelAt(src, x + dx, y + dy, n);

    int side = 2 * radius + 1;
    dst[y * n + x] = toByte((float)sum / (float)(side * side));
}

void box3x3_gpu(GpuBuffers& g) {
    boxBlurKernel<<<gridFor(g.n), blockFor()>>>(g.src, g.dst, g.n, 1);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
}

void box5x5_gpu(GpuBuffers& g) {
    boxBlurKernel<<<gridFor(g.n), blockFor()>>>(g.src, g.dst, g.n, 2);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
}
