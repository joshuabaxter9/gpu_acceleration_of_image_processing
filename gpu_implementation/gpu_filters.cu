// gpu_filters.cu
// Device buffer management, host/device transfers, and the GPU test table.

#include "gpu_filters.h"
#include "gpu_common.cuh"

// ---------------------------------------------------------------------------
// Buffers
// ---------------------------------------------------------------------------

GpuBuffers makeGpuBuffers(const cv::Mat& src) {
    GpuBuffers g;
    g.n = src.cols;
    size_t bytes8  = (size_t)g.n * g.n * sizeof(unsigned char);
    size_t bytes16 = (size_t)g.n * g.n * sizeof(short);

    cudaCheck(cudaMalloc(&g.src, bytes8));
    cudaCheck(cudaMalloc(&g.dst, bytes8));
    cudaCheck(cudaMalloc(&g.tmp, bytes8));
    cudaCheck(cudaMalloc(&g.gx,  bytes16));
    cudaCheck(cudaMalloc(&g.gy,  bytes16));

    g.hostSrc = src;
    g.hostDst.create(src.size(), CV_8UC1);
    g.hostGx.create(src.size(), CV_16SC1);
    g.hostGy.create(src.size(), CV_16SC1);

    cudaCheck(cudaMemcpy(g.src, src.data, bytes8, cudaMemcpyHostToDevice));
    return g;
}

void freeGpuBuffers(GpuBuffers& g) {
    cudaCheck(cudaFree(g.src));
    cudaCheck(cudaFree(g.dst));
    cudaCheck(cudaFree(g.tmp));
    cudaCheck(cudaFree(g.gx));
    cudaCheck(cudaFree(g.gy));
    g.src = g.dst = g.tmp = nullptr;
    g.gx = g.gy = nullptr;
}

void downloadOutputs(GpuBuffers& g) {
    size_t bytes8  = (size_t)g.n * g.n * sizeof(unsigned char);
    size_t bytes16 = (size_t)g.n * g.n * sizeof(short);
    cudaCheck(cudaMemcpy(g.hostDst.data, g.dst, bytes8,  cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(g.hostGx.data,  g.gx,  bytes16, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(g.hostGy.data,  g.gy,  bytes16, cudaMemcpyDeviceToHost));
}

// ---------------------------------------------------------------------------
// Transfer cost
// ---------------------------------------------------------------------------

void transferRoundTrip(GpuBuffers& g) {
    size_t bytes8 = (size_t)g.n * g.n * sizeof(unsigned char);
    cudaCheck(cudaMemcpy(g.src, g.hostSrc.data, bytes8, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(g.hostDst.data, g.dst, bytes8, cudaMemcpyDeviceToHost));
}

// ---------------------------------------------------------------------------
// Test table
// ---------------------------------------------------------------------------

const GpuTest kGpuTests[] = {
    {"box3x3",    box3x3_gpu},
    {"box5x5",    box5x5_gpu},
    {"gauss5x5",  gauss5x5_gpu},
    {"median3x3", median3x3_gpu},
    {"sobel3x3",  sobel3x3_gpu},
    {"unsharp",   unsharp_gpu},
    {"transfer",  transferRoundTrip},
};

const int kGpuTestCount = sizeof(kGpuTests) / sizeof(kGpuTests[0]);
