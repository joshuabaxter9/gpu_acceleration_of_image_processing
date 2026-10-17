// 2d_gaussian_blur.cu
// 5x5 Gaussian blur, sigma = 1.0, done as a full 2D convolution (25 taps per
// pixel). One thread per output pixel, reading straight from global memory.

#include "gpu_filters.h"

__device__ int clampIndexGauss(int i, int n)
{
    if (i < 0) return 0;
    if (i >= n) return n - 1;
    return i;
}

__global__ void gauss5x5Kernel(const unsigned char *src, unsigned char *dst, int n)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= n || y >= n) return;

    // 1D weights for sigma = 1.0, size 5 (same values OpenCV uses). The 2D
    // weight of tap (dx, dy) is w[dx] * w[dy].
    const float w[5] = {0.0545f, 0.2442f, 0.4026f, 0.2442f, 0.0545f};

    float sum = 0.0f;
    for (int dy = -2; dy <= 2; ++dy)
    {
        for (int dx = -2; dx <= 2; ++dx)
        {
            int sx = clampIndexGauss(x + dx, n);
            int sy = clampIndexGauss(y + dy, n);
            sum += w[dy + 2] * w[dx + 2] * src[sy * n + sx];
        }
    }

    dst[y * n + x] = (unsigned char)(sum + 0.5f);
}

void launchGauss5x5(const unsigned char *src, unsigned char *dst, int n)
{
    dim3 block(kBlockSize, kBlockSize);
    dim3 grid((n + kBlockSize - 1) / kBlockSize, (n + kBlockSize - 1) / kBlockSize);
    gauss5x5Kernel<<<grid, block>>>(src, dst, n);
}

void gauss5x5_gpu(GpuBuffers &g)
{
    launchGauss5x5(g.src, g.dst, g.n);
    cudaDeviceSynchronize();
}
