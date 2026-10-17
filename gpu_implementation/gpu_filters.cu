// gpu_filters.cu
// Device memory setup and the GPU test table.

#include "gpu_filters.h"

GpuBuffers makeGpuBuffers(const cv::Mat &src)
{
    GpuBuffers g;
    g.n = src.cols;

    size_t bytes8 = (size_t)g.n * g.n * sizeof(unsigned char);
    size_t bytes16 = (size_t)g.n * g.n * sizeof(short);

    cudaMalloc(&g.src, bytes8);
    cudaMalloc(&g.dst, bytes8);
    cudaMalloc(&g.tmp, bytes8);
    cudaMalloc(&g.gx, bytes16);
    cudaMalloc(&g.gy, bytes16);

    cudaMemcpy(g.src, src.data, bytes8, cudaMemcpyHostToDevice);
    return g;
}

void freeGpuBuffers(GpuBuffers &g)
{
    cudaFree(g.src);
    cudaFree(g.dst);
    cudaFree(g.tmp);
    cudaFree(g.gx);
    cudaFree(g.gy);
}

const GpuTest kGpuTests[] = {
    {"box3x3", box3x3_gpu},
    {"box5x5", box5x5_gpu},
    {"gauss5x5", gauss5x5_gpu},
    {"median3x3", median3x3_gpu},
    {"sobel3x3", sobel3x3_gpu},
    {"unsharp", unsharp_gpu},
};

const int kGpuTestCount = sizeof(kGpuTests) / sizeof(kGpuTests[0]);
