// 2d_gaussian_blur.cu
// 5x5 Gaussian blur, sigma = 1.0, done as a full 2D convolution (25 taps per
// pixel). OpenCV's CPU version runs two separable 1D passes instead; note
// that difference in the write-up.

#include "gpu_filters.h"
#include "gpu_common.cuh"

// 1D Gaussian weights for size 5, sigma 1.0, matching cv::getGaussianKernel.
// The 2D weight for tap (i, j) is kGauss1D[i] * kGauss1D[j].
__constant__ float kGauss1D[5] = {
    0.054488685f, 0.244201342f, 0.402619947f, 0.244201342f, 0.054488685f
};

__global__ void gaussBlurKernel(const unsigned char* src, unsigned char* dst,
                                int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= n || y >= n) return;

    float sum = 0.0f;
    for (int dy = -2; dy <= 2; ++dy)
        for (int dx = -2; dx <= 2; ++dx)
            sum += kGauss1D[dy + 2] * kGauss1D[dx + 2]
                 * pixelAt(src, x + dx, y + dy, n);

    dst[y * n + x] = toByte(sum);
}

void launchGauss5x5(const unsigned char* src, unsigned char* dst, int n) {
    gaussBlurKernel<<<gridFor(n), blockFor()>>>(src, dst, n);
    cudaCheck(cudaGetLastError());
}

void gauss5x5_gpu(GpuBuffers& g) {
    launchGauss5x5(g.src, g.dst, g.n);
    cudaCheck(cudaDeviceSynchronize());
}
