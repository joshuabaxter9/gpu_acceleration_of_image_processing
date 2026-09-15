// gpu_filters.h
// CUDA implementations of every filter in the benchmark. Mirrors
// cpu_implementation/cpu_filters.h: same filter names, same table layout,
// so main.cu can time CPU and GPU with the same loop.

#pragma once

#include <opencv2/opencv.hpp>

// All device memory a filter may read or write, plus host-side mats for
// uploading input and downloading results. Allocated once per image size in
// main.cu, outside the timed region, and reused for every iteration.
struct GpuBuffers {
    int n;               // image is n x n

    unsigned char* src;  // device input, 8-bit grayscale
    unsigned char* dst;  // device output, 8-bit
    unsigned char* tmp;  // device scratch, 8-bit (used by unsharp masking)
    short*         gx;   // device horizontal gradient, 16-bit signed (Sobel)
    short*         gy;   // device vertical gradient, 16-bit signed (Sobel)

    cv::Mat hostSrc;     // the input image on the host (for uploads)
    cv::Mat hostDst;     // download targets for verification
    cv::Mat hostGx;
    cv::Mat hostGy;
};

// Every benchmarked GPU filter has this shape. Each wrapper launches its
// kernel(s) and ends with cudaDeviceSynchronize() so the caller's timer sees
// the whole kernel, not just the launch.
typedef void (*GpuFilterFn)(GpuBuffers&);

// A row in the GPU test table.
struct GpuTest {
    const char* name;
    GpuFilterFn fn;
};

// Allocate device buffers for src and upload it. Call freeGpuBuffers after.
GpuBuffers makeGpuBuffers(const cv::Mat& src);
void       freeGpuBuffers(GpuBuffers& g);

// Copy dst, gx, gy from the device into the host mats (for verification).
void downloadOutputs(GpuBuffers& g);

// Times one host-to-device upload of src plus one device-to-host download
// of dst. Listed in the table so transfer overhead appears in results.csv.
void transferRoundTrip(GpuBuffers& g);

// The GPU filters. Same names as the CPU side with a _gpu suffix.
void box3x3_gpu(GpuBuffers& g);     // box_blur.cu
void box5x5_gpu(GpuBuffers& g);     // box_blur.cu
void gauss5x5_gpu(GpuBuffers& g);   // 2d_gaussian_blur.cu
void median3x3_gpu(GpuBuffers& g);  // median_filter.cu
void sobel3x3_gpu(GpuBuffers& g);   // sobel_filter.cu
void unsharp_gpu(GpuBuffers& g);    // unsharp_masking.cu

// Host-side launcher for the Gaussian kernel, shared with unsharp masking.
void launchGauss5x5(const unsigned char* src, unsigned char* dst, int n);

// The table main.cu iterates over.
extern const GpuTest kGpuTests[];
extern const int     kGpuTestCount;
