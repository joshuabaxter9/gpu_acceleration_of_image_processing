// cpu_filters.h
// OpenCV reference implementations of every filter in the benchmark.
// Each filter has the same signature so main.cu can time them from a table.

#pragma once

#include <opencv2/opencv.hpp>

// Border handling used by every CPU filter. The CUDA kernels should use the
// same rule (clamp to edge) so CPU and GPU outputs can be compared directly.
static const int kBorder = cv::BORDER_REPLICATE;

// All images a filter may read or write. Allocated once per image size in
// main.cu, outside the timed region, and reused for every iteration.
struct Buffers {
    cv::Mat src;   // input, 8-bit grayscale, N x N
    cv::Mat dst;   // output, same type as src
    cv::Mat tmp;   // scratch, same type as src (used by unsharp masking)
    cv::Mat gx;    // horizontal gradient, 16-bit signed (used by Sobel)
    cv::Mat gy;    // vertical gradient, 16-bit signed (used by Sobel)
};

// Every benchmarked filter has this shape.
typedef void (*FilterFn)(Buffers&);

// A row in the test table: a label for the CSV and the function to time.
struct Test {
    const char* name;
    FilterFn    fn;
};

// Build a Buffers set for an N x N version of the base image.
Buffers makeBuffers(const cv::Mat& base, int n);

// The CPU filters.
void box3x3(Buffers& b);
void box5x5(Buffers& b);
void gauss5x5(Buffers& b);
void median3x3(Buffers& b);
void sobel3x3(Buffers& b);
void unsharp(Buffers& b);

// The table main.cu iterates over.
extern const Test kCpuTests[];
extern const int  kCpuTestCount;
