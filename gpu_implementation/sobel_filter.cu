// sobel_filter.cu
// 3x3 Sobel edge detector. One thread per pixel computes both the horizontal
// gradient (gx) and vertical gradient (gy). Output is 16-bit signed so
// negative gradients are not clipped, matching the CPU side.

#include "gpu_filters.h"

__device__ int clampIndexSobel(int i, int n)
{
    if (i < 0) return 0;
    if (i >= n) return n - 1;
    return i;
}

__global__ void sobel3x3Kernel(const unsigned char *src, short *gx, short *gy, int n)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= n || y >= n) return;

    // Load the 3x3 neighborhood: p[row][col].
    int p[3][3];
    for (int dy = -1; dy <= 1; ++dy)
    {
        for (int dx = -1; dx <= 1; ++dx)
        {
            int sx = clampIndexSobel(x + dx, n);
            int sy = clampIndexSobel(y + dy, n);
            p[dy + 1][dx + 1] = src[sy * n + sx];
        }
    }

    // Horizontal:  [-1 0 1]        Vertical:  [-1 -2 -1]
    //              [-2 0 2]                   [ 0  0  0]
    //              [-1 0 1]                   [ 1  2  1]
    int sx = (p[0][2] - p[0][0]) + 2 * (p[1][2] - p[1][0]) + (p[2][2] - p[2][0]);
    int sy = (p[2][0] - p[0][0]) + 2 * (p[2][1] - p[0][1]) + (p[2][2] - p[0][2]);

    gx[y * n + x] = (short)sx;
    gy[y * n + x] = (short)sy;
}

void sobel3x3_gpu(GpuBuffers &g)
{
    dim3 block(kBlockSize, kBlockSize);
    dim3 grid((g.n + kBlockSize - 1) / kBlockSize, (g.n + kBlockSize - 1) / kBlockSize);
    sobel3x3Kernel<<<grid, block>>>(g.src, g.gx, g.gy, g.n);
    cudaDeviceSynchronize();
}
