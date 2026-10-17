// cpu_filters.h

#pragma once

#include <opencv2/opencv.hpp>

static const int kBorder = cv::BORDER_REPLICATE;

struct Buffers
{
    cv::Mat src; // input, 8-bit grayscale, N x N
    cv::Mat dst; // output, same type as src
    cv::Mat tmp; // scratch, same type as src (used by unsharp masking)
    cv::Mat gx;  // horizontal gradient, 16-bit signed (used by Sobel)
    cv::Mat gy;  // vertical gradient, 16-bit signed (used by Sobel)
};

typedef void (*FilterFn)(Buffers &);

struct Test
{
    const char *name;
    FilterFn fn;
};

Buffers makeBuffers(const cv::Mat &base, int n);

void box3x3(Buffers &b);
void box5x5(Buffers &b);
void gauss5x5(Buffers &b);
void median3x3(Buffers &b);
void sobel3x3(Buffers &b);
void unsharp(Buffers &b);

extern const Test kCpuTests[];
extern const int kCpuTestCount;
