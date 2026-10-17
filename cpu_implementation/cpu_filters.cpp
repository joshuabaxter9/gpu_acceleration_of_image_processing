// cpu_filters.cpp

#include "cpu_filters.h"

Buffers makeBuffers(const cv::Mat &base, int n)
{
    Buffers b;
    int interp = (n < base.cols) ? cv::INTER_AREA : cv::INTER_CUBIC;
    cv::resize(base, b.src, cv::Size(n, n), 0, 0, interp);

    b.dst.create(b.src.size(), b.src.type());
    b.tmp.create(b.src.size(), b.src.type());
    b.gx.create(b.src.size(), CV_16SC1);
    b.gy.create(b.src.size(), CV_16SC1);
    return b;
}

// Normalized box blur: each output pixel is the mean of its 3x3 neighborhood.
void box3x3(Buffers &b)
{
    cv::blur(b.src, b.dst, cv::Size(3, 3), cv::Point(-1, -1), kBorder);
}

// Same as box3x3 with a 5x5 neighborhood.
void box5x5(Buffers &b)
{
    cv::blur(b.src, b.dst, cv::Size(5, 5), cv::Point(-1, -1), kBorder);
}

// 5x5 Gaussian blur, sigma = 1.0 in both directions.
// Note: OpenCV runs this as two separable 1D passes, not a full 2D convolution.
void gauss5x5(Buffers &b)
{
    cv::GaussianBlur(b.src, b.dst, cv::Size(5, 5), 1.0, 1.0, kBorder);
}

// 3x3 median filter. OpenCV has no border argument for medianBlur; it
// replicates the edge internally, which matches kBorder.
void median3x3(Buffers &b)
{
    cv::medianBlur(b.src, b.dst, 3);
}

// 3x3 Sobel edge detector: horizontal gradient (dx=1) and vertical gradient
// (dy=1). Output is 16-bit signed so negative gradients are not clipped.
void sobel3x3(Buffers &b)
{
    cv::Sobel(b.src, b.gx, CV_16S, 1, 0, 3, 1, 0, kBorder);
    cv::Sobel(b.src, b.gy, CV_16S, 0, 1, 3, 1, 0, kBorder);
}

// Unsharp masking: dst = src + amount * (src - blur(src)), with amount = 1.0.
// Written as a single addWeighted: dst = 2*src - 1*blur.
void unsharp(Buffers &b)
{
    cv::GaussianBlur(b.src, b.tmp, cv::Size(5, 5), 1.0, 1.0, kBorder);
    cv::addWeighted(b.src, 2.0, b.tmp, -1.0, 0, b.dst);
}

const Test kCpuTests[] = {
    {"box3x3", box3x3},
    {"box5x5", box5x5},
    {"gauss5x5", gauss5x5},
    {"median3x3", median3x3},
    {"sobel3x3", sobel3x3},
    {"unsharp", unsharp},
};

const int kCpuTestCount = sizeof(kCpuTests) / sizeof(kCpuTests[0]);
