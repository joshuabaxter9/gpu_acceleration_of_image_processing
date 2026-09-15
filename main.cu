// main.cu will run all benchmarks and be in charge or handling logging the results

// cpu algorithm library inclusion
#include "cpu_implementation/cpu_filters.h"

// gpu algorithm library inclusion
#include "gpu_implementation/gpu_filters.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Benchmark configuration
// ---------------------------------------------------------------------------

// Source photo. Any format OpenCV can decode (jpeg, png, bmp, ...).
static const std::string kImagePath = "test_images/galaxy.jpeg";

// Square image sizes (N x N) to sweep. Small sizes show GPU launch/transfer
// overhead, large sizes show raw throughput.
static const std::vector<int> kSizes = {256, 512, 1024, 2048, 4096, 8192};

// Where results are written, in CSV form.
static const std::string kResultsPath = "results.csv";

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

// Runs fn once untimed (warm-up), then times `iters` calls.
// Returns mean milliseconds per call.
double benchmark(FilterFn fn, Buffers& b, int iters) {
    fn(b);  // warm-up: thread pool start, page faults, cache fill, etc.

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) fn(b);
    auto end = std::chrono::high_resolution_clock::now();

    double total = std::chrono::duration<double, std::milli>(end - start).count();
    return total / iters;
}

// Same thing for a GPU filter. Every GPU wrapper ends with
// cudaDeviceSynchronize(), so the timer covers the whole kernel.
double benchmark(GpuFilterFn fn, GpuBuffers& g, int iters) {
    fn(g);  // warm-up: CUDA context creation, module load, first launch

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) fn(g);
    auto end = std::chrono::high_resolution_clock::now();

    double total = std::chrono::duration<double, std::milli>(end - start).count();
    return total / iters;
}

// Fewer iterations for big images so the whole sweep finishes in reasonable time.
int itersForSize(int n) {
    if (n <= 512)  return 500;
    if (n <= 2048) return 100;
    if (n <= 4096) return 30;
    return 10;
}

// ---------------------------------------------------------------------------
// Logging
// ---------------------------------------------------------------------------

static const std::string kCsvHeader = "filter,device,size,iters,mean_ms,mpix_per_s";

// Writes one result row to both the CSV file and the terminal.
void writeRow(std::ofstream& out, const char* filter, const char* device,
              int n, int iters, double ms) {
    double mpix = static_cast<double>(n) * n / 1e6;
    double throughput = mpix / (ms / 1000.0);

    out       << filter << "," << device << "," << n << "," << iters << ","
              << ms << "," << throughput << "\n";
    std::cout << filter << "," << device << "," << n << "," << iters << ","
              << ms << "," << throughput << "\n";
}

// ---------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------

// Largest absolute pixel difference between two images of the same type.
double maxAbsDiff(const cv::Mat& a, const cv::Mat& b) {
    cv::Mat diff;
    cv::absdiff(a, b, diff);
    double maxVal = 0.0;
    cv::minMaxLoc(diff, nullptr, &maxVal);
    return maxVal;
}

// Runs each CPU filter and its GPU counterpart once and prints the largest
// pixel difference. Small differences (1 or 2) come from rounding; anything
// large means a kernel is wrong. Not timed.
void verify(Buffers& b, GpuBuffers& g, int n) {
    std::cout << "verify " << n << "x" << n << ":";

    for (int i = 0; i < kCpuTestCount; ++i) {
        const Test& cpu = kCpuTests[i];

        // Find the GPU test with the same name.
        const GpuTest* gpu = nullptr;
        for (int j = 0; j < kGpuTestCount; ++j)
            if (std::string(kGpuTests[j].name) == cpu.name) gpu = &kGpuTests[j];
        if (!gpu) continue;

        cpu.fn(b);
        gpu->fn(g);
        downloadOutputs(g);

        double diff;
        if (std::string(cpu.name) == "sobel3x3")
            diff = std::max(maxAbsDiff(b.gx, g.hostGx), maxAbsDiff(b.gy, g.hostGy));
        else
            diff = maxAbsDiff(b.dst, g.hostDst);

        std::cout << " " << cpu.name << "=" << diff;
    }
    std::cout << "\n";
}

// ---------------------------------------------------------------------------
// GPU detection
// ---------------------------------------------------------------------------

// Returns true if a CUDA-capable GPU is present and prints its name.
// Otherwise explains what was found and asks the user whether to continue
// with the CPU tests alone. Returns false to run CPU-only, or exits if the
// user declines.
bool detectGpu() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err == cudaSuccess && deviceCount > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "GPU: " << prop.name
                  << " (sm_" << prop.major << prop.minor << ", "
                  << prop.totalGlobalMem / (1024 * 1024) << " MiB)\n";
        return true;
    }

    std::cout << "No CUDA-capable GPU was detected";
    if (err != cudaSuccess)
        std::cout << " (" << cudaGetErrorString(err) << ")";
    std::cout << ".\n";

    std::cout << "Run the CPU tests only? [y/n] ";
    std::string answer;
    std::getline(std::cin, answer);

    if (!answer.empty() && (answer[0] == 'y' || answer[0] == 'Y')) {
        std::cout << "Running CPU tests only.\n";
        return false;
    }

    std::cout << "Exiting.\n";
    std::exit(0);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    cv::Mat base = cv::imread(kImagePath, cv::IMREAD_GRAYSCALE);
    if (base.empty()) {
        std::cerr << "failed to load image: " << kImagePath << "\n";
        return 1;
    }

    std::ofstream results(kResultsPath);
    if (!results) {
        std::cerr << "failed to open " << kResultsPath << " for writing\n";
        return 1;
    }

    std::cout << "OpenCV " << CV_VERSION
              << ", threads = " << cv::getNumThreads()
              << ", optimized = " << (cv::useOptimized() ? "yes" : "no") << "\n";
    std::cout << "source image: " << kImagePath
              << " (" << base.cols << "x" << base.rows << ")\n";

    bool haveGpu = detectGpu();
    std::cout << "\n";

    results   << kCsvHeader << "\n";
    std::cout << kCsvHeader << "\n";

    for (int n : kSizes) {
        // Setup is NOT timed: resize the photo and allocate all buffers once.
        Buffers b = makeBuffers(base, n);
        int iters = itersForSize(n);

        /* cpu tests start here */
        for (int i = 0; i < kCpuTestCount; ++i) {
            const Test& t = kCpuTests[i];
            double ms = benchmark(t.fn, b, iters);
            writeRow(results, t.name, "cpu", n, iters, ms);
        }

        if (!haveGpu) continue;

        /* gpu tests start here. Input is uploaded once in makeGpuBuffers, so
           the filter rows are kernel-only timings; the "transfer" row shows
           the cost of one upload plus one download at this size. */
        GpuBuffers g = makeGpuBuffers(b.src);

        for (int i = 0; i < kGpuTestCount; ++i) {
            const GpuTest& t = kGpuTests[i];
            double ms = benchmark(t.fn, g, iters);
            writeRow(results, t.name, "gpu", n, iters, ms);
        }

        verify(b, g, n);
        freeGpuBuffers(g);
    }

    std::cout << "\nresults written to " << kResultsPath << "\n";
    return 0;
}
