// box_blur.cu
// Normalized box blur. One thread per output pixel, reading straight from
// global memory. Edge pixels are handled by clamping the coordinate to the
// image (same as cv::BORDER_REPLICATE on the CPU side).

#include "gpu_filters.h"

__device__ int clampIndex(int i, int n)
{
    if (i < 0) return 0;
    if (i >= n) return n - 1;
    return i;
}

__global__ void boxBlurKernel(const unsigned char *src, unsigned char *dst,
                              int n, int radius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= n || y >= n) return;

    int sum = 0;
    for (int dy = -radius; dy <= radius; ++dy)
    {
        for (int dx = -radius; dx <= radius; ++dx)
        {
            int sx = clampIndex(x + dx, n);
            int sy = clampIndex(y + dy, n);
            sum += src[sy * n + sx];
        }
    }

    int side = 2 * radius + 1;
    dst[y * n + x] = (unsigned char)((sum + side * side / 2) / (side * side));
}

void box3x3_gpu(GpuBuffers &g)
{
    dim3 block(kBlockSize, kBlockSize);
    dim3 grid((g.n + kBlockSize - 1) / kBlockSize, (g.n + kBlockSize - 1) / kBlockSize);
    boxBlurKernel<<<grid, block>>>(g.src, g.dst, g.n, 1);
    cudaDeviceSynchronize();
}

void box5x5_gpu(GpuBuffers &g)
{
    dim3 block(kBlockSize, kBlockSize);
    dim3 grid((g.n + kBlockSize - 1) / kBlockSize, (g.n + kBlockSize - 1) / kBlockSize);
    boxBlurKernel<<<grid, block>>>(g.src, g.dst, g.n, 2);
    cudaDeviceSynchronize();
}
