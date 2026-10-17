// gpu_filters.h
// Mirrors cpu_filters.h so main.cu can time CPU and GPU with the same loop.

#pragma once

#include <opencv2/opencv.hpp>

struct GpuBuffers
{
    int n;               // image is n x n

    unsigned char *src;  // device input, 8-bit grayscale
    unsigned char *dst;  // device output, 8-bit
    unsigned char *tmp;  // device scratch, 8-bit (used by unsharp masking)
    short *gx;           // device horizontal gradient, 16-bit signed (Sobel)
    short *gy;           // device vertical gradient, 16-bit signed (Sobel)
};

// Every wrapper launches its kernel and ends with cudaDeviceSynchronize(),
// so the host timer in main.cu sees the whole kernel, not just the launch.
typedef void (*GpuFilterFn)(GpuBuffers &);

struct GpuTest
{
    const char *name;
    GpuFilterFn fn;
};

// Threads per block along each axis (16 x 16 = 256 threads).
static const int kBlockSize = 16;

// Allocate device buffers and upload src. Call freeGpuBuffers when done.
GpuBuffers makeGpuBuffers(const cv::Mat &src);
void freeGpuBuffers(GpuBuffers &g);

void box3x3_gpu(GpuBuffers &g);    // box_blur.cu
void box5x5_gpu(GpuBuffers &g);    // box_blur.cu
void gauss5x5_gpu(GpuBuffers &g);  // 2d_gaussian_blur.cu
void median3x3_gpu(GpuBuffers &g); // median_filter.cu
void sobel3x3_gpu(GpuBuffers &g);  // sobel_filter.cu
void unsharp_gpu(GpuBuffers &g);   // unsharp_masking.cu

// Launches the Gaussian kernel on any src/dst pair. Shared with unsharp masking.
void launchGauss5x5(const unsigned char *src, unsigned char *dst, int n);

extern const GpuTest kGpuTests[];
extern const int kGpuTestCount;
