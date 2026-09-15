// main.cu will run all benchmarks and be in charge or handling logging the results

// cpu algorithm library inclusion
#include "cpu_implementation/cpu_filters.h"

// gpu algorithm library inclusion
// When the GPU side is added, include its header here:
//   #include "gpu_implementation/gpu_filters.h"
// That header should mirror cpu_filters.h: a GpuBuffers struct holding device
// pointers (src, dst, tmp, gx, gy) plus host mats for download, a
// GpuFilterFn typedef (void (*)(GpuBuffers&)), a GpuTest {name, fn} row, and
// an extern kGpuTests[] table. Each gpu_implementation/*.cu file defines one
// kernel and a wrapper that launches it and ends with cudaDeviceSynchronize().

#include <chrono>
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

// GPU timing will be a second overload with the same body:
//   double benchmark(GpuFilterFn fn, GpuBuffers& g, int iters);
// Because every GPU wrapper ends with cudaDeviceSynchronize(), the host
// timer covers the whole kernel and no cudaEvent code is needed here.

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
// The device column is "cpu" now and will also take "gpu" rows later, so
// both implementations land in the same file for plotting.
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
// GPU detection (to be added with the GPU side)
// ---------------------------------------------------------------------------

// A detectGpu() function will call cudaGetDeviceCount(). If a device is
// found it prints the name and compute capability and returns true. If not,
// it prints the reason, asks "Run the CPU tests only? [y/n]", and returns
// false or exits. main() stores the result in a bool and skips the GPU
// block of the size loop when it is false.

// ---------------------------------------------------------------------------
// Verification (to be added with the GPU side)
// ---------------------------------------------------------------------------

// A verify(Buffers&, GpuBuffers&, int n) function will run each CPU filter
// and the GPU filter of the same name once, download the GPU result, and
// print the largest absolute pixel difference using cv::absdiff and
// cv::minMaxLoc. Box, median, and Sobel should match exactly; Gaussian and
// unsharp may differ by 1 because OpenCV uses fixed-point weights.

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

    // GPU side: bool haveGpu = detectGpu();
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

        /* gpu tests will go here, using the same shape as the CPU loop:
         *
         *   if (!haveGpu) continue;
         *   GpuBuffers g = makeGpuBuffers(b.src);   // cudaMalloc + upload src once
         *   for (int i = 0; i < kGpuTestCount; ++i) {
         *       const GpuTest& t = kGpuTests[i];
         *       double ms = benchmark(t.fn, g, iters);
         *       writeRow(results, t.name, "gpu", n, iters, ms);
         *   }
         *   verify(b, g, n);
         *   freeGpuBuffers(g);
         *
         * Input is uploaded once outside the timed loop, so filter rows are
         * kernel-only timings. A "transfer" entry in kGpuTests that does one
         * upload plus one download records the PCIe cost separately.
         */
    }

    std::cout << "\nresults written to " << kResultsPath << "\n";
    return 0;
}
