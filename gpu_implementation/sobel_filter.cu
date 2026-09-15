// sobel_filter.cu
// 3x3 Sobel edge detector. Writes the horizontal gradient to gx and the
// vertical gradient to gy as 16-bit signed values, matching cv::Sobel with
// CV_16S output on the CPU side.
//
//   Gx = [-1 0 1]      Gy = [-1 -2 -1]
//        [-2 0 2]           [ 0  0  0]
//        [-1 0 1]           [ 1  2  1]

#include "gpu_filters.h"
#include "gpu_common.cuh"

__global__ void sobelKernel(const unsigned char* src, short* gx, short* gy,
                            int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= n || y >= n) return;

    int p00 = pixelAt(src, x - 1, y - 1, n);
    int p01 = pixelAt(src, x,     y - 1, n);
    int p02 = pixelAt(src, x + 1, y - 1, n);
    int p10 = pixelAt(src, x - 1, y,     n);
    int p12 = pixelAt(src, x + 1, y,     n);
    int p20 = pixelAt(src, x - 1, y + 1, n);
    int p21 = pixelAt(src, x,     y + 1, n);
    int p22 = pixelAt(src, x + 1, y + 1, n);

    int sx = (p02 + 2 * p12 + p22) - (p00 + 2 * p10 + p20);
    int sy = (p20 + 2 * p21 + p22) - (p00 + 2 * p01 + p02);

    gx[y * n + x] = (short)sx;
    gy[y * n + x] = (short)sy;
}

void sobel3x3_gpu(GpuBuffers& g) {
    sobelKernel<<<gridFor(g.n), blockFor()>>>(g.src, g.gx, g.gy, g.n);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
}
