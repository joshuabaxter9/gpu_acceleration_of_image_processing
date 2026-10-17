// unsharp_masking.cu
// Unsharp masking: dst = src + amount * (src - blur(src)), with amount = 1.0,
// so dst = 2*src - blur. Two launches: the Gaussian kernel into tmp, then a
// per-pixel combine kernel.

#include "gpu_filters.h"

__global__ void unsharpCombineKernel(const unsigned char *src, const unsigned char *blur,
                                     unsigned char *dst, int n)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= n || y >= n) return;

    int i = y * n + x;
    int v = 2 * src[i] - blur[i];
    if (v < 0) v = 0;
    if (v > 255) v = 255;
    dst[i] = (unsigned char)v;
}

void unsharp_gpu(GpuBuffers &g)
{
    launchGauss5x5(g.src, g.tmp, g.n);

    dim3 block(kBlockSize, kBlockSize);
    dim3 grid((g.n + kBlockSize - 1) / kBlockSize, (g.n + kBlockSize - 1) / kBlockSize);
    unsharpCombineKernel<<<grid, block>>>(g.src, g.tmp, g.dst, g.n);
    cudaDeviceSynchronize();
}
