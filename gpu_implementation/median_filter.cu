// median_filter.cu
// 3x3 median filter. Each thread loads its 9 neighbors into a local array,
// bubble sorts them, and writes the middle value.

#include "gpu_filters.h"

__device__ int clampIndexMedian(int i, int n)
{
    if (i < 0) return 0;
    if (i >= n) return n - 1;
    return i;
}

__global__ void median3x3Kernel(const unsigned char *src, unsigned char *dst, int n)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= n || y >= n) return;

    unsigned char v[9];
    int k = 0;
    for (int dy = -1; dy <= 1; ++dy)
    {
        for (int dx = -1; dx <= 1; ++dx)
        {
            int sx = clampIndexMedian(x + dx, n);
            int sy = clampIndexMedian(y + dy, n);
            v[k++] = src[sy * n + sx];
        }
    }

    // Bubble sort the 9 values.
    for (int i = 0; i < 9; ++i)
    {
        for (int j = 0; j < 8 - i; ++j)
        {
            if (v[j] > v[j + 1])
            {
                unsigned char t = v[j];
                v[j] = v[j + 1];
                v[j + 1] = t;
            }
        }
    }

    dst[y * n + x] = v[4];
}

void median3x3_gpu(GpuBuffers &g)
{
    dim3 block(kBlockSize, kBlockSize);
    dim3 grid((g.n + kBlockSize - 1) / kBlockSize, (g.n + kBlockSize - 1) / kBlockSize);
    median3x3Kernel<<<grid, block>>>(g.src, g.dst, g.n);
    cudaDeviceSynchronize();
}
