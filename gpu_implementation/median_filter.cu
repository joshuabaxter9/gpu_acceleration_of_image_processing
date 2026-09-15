// median_filter.cu
// 3x3 median filter. Each thread loads its 9 neighbors into registers and
// sorts them with a fixed compare-swap network, then takes the middle value.

#include "gpu_filters.h"
#include "gpu_common.cuh"

__device__ __forceinline__ void sortPair(unsigned char& a, unsigned char& b) {
    unsigned char lo = min(a, b);
    unsigned char hi = max(a, b);
    a = lo;
    b = hi;
}

__global__ void medianKernel(const unsigned char* src, unsigned char* dst,
                             int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= n || y >= n) return;

    unsigned char v[9];
    int k = 0;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
            v[k++] = pixelAt(src, x + dx, y + dy, n);

    // 19-comparator network that leaves the median in v[4].
    sortPair(v[1], v[2]); sortPair(v[4], v[5]); sortPair(v[7], v[8]);
    sortPair(v[0], v[1]); sortPair(v[3], v[4]); sortPair(v[6], v[7]);
    sortPair(v[1], v[2]); sortPair(v[4], v[5]); sortPair(v[7], v[8]);
    sortPair(v[0], v[3]); sortPair(v[5], v[8]); sortPair(v[4], v[7]);
    sortPair(v[3], v[6]); sortPair(v[1], v[4]); sortPair(v[2], v[5]);
    sortPair(v[4], v[7]); sortPair(v[4], v[2]); sortPair(v[6], v[4]);
    sortPair(v[4], v[2]);

    dst[y * n + x] = v[4];
}

void median3x3_gpu(GpuBuffers& g) {
    medianKernel<<<gridFor(g.n), blockFor()>>>(g.src, g.dst, g.n);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
}
