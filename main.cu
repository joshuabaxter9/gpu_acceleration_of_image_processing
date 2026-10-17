// main.cu will run all benchmarks and be in charge or handling logging the results

// cpu algorithm library inclusion
#include "cpu_implementation/cpu_filters.h"

// gpu algorithm library inclusion
#include "gpu_implementation/gpu_filters.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// Every readable image in this directory is benchmarked, in sorted name order.
static const std::string kImageDir = "test_images";

static const std::vector<int> kSizes = {256, 512, 1024, 2048, 4096, 8192};

// Where results are written, in CSV form.
static const std::string kResultsPath = "results.csv";

// Summary of one test's per-iteration runtimes, all in milliseconds.
struct Stats {
    double mean;
    double median;
    double stddev;  // sample standard deviation (divides by n - 1)
};

// Mean, median and standard deviation of the recorded iteration times.
Stats summarize(std::vector<double> times) {
    Stats s;
    size_t n = times.size();

    double sum = 0.0;
    for (double t : times) sum += t;
    s.mean = sum / n;

    std::sort(times.begin(), times.end());
    if (n % 2 == 1) s.median = times[n / 2];
    else            s.median = (times[n / 2 - 1] + times[n / 2]) / 2.0;

    double sqDiff = 0.0;
    for (double t : times) sqDiff += (t - s.mean) * (t - s.mean);
    s.stddev = (n > 1) ? std::sqrt(sqDiff / (n - 1)) : 0.0;

    return s;
}

// Each iteration is timed on its own so the median and spread can be
// reported, not just the average.
Stats benchmark(FilterFn fn, Buffers& b, int iters) {
    fn(b);  // warm-up: thread pool start, page faults, cache fill, etc.

    std::vector<double> times(iters);
    for (int i = 0; i < iters; ++i) {
        auto start = std::chrono::steady_clock::now();
        fn(b);
        auto end = std::chrono::steady_clock::now();
        times[i] = std::chrono::duration<double, std::milli>(end - start).count();
    }
    return summarize(times);
}

// Same body for the GPU side. Every GPU wrapper ends with
// cudaDeviceSynchronize(), so the host timer covers the whole kernel.
Stats benchmark(GpuFilterFn fn, GpuBuffers& g, int iters) {
    fn(g);  // warm-up: first launch pays module load / context setup

    std::vector<double> times(iters);
    for (int i = 0; i < iters; ++i) {
        auto start = std::chrono::steady_clock::now();
        fn(g);
        auto end = std::chrono::steady_clock::now();
        times[i] = std::chrono::duration<double, std::milli>(end - start).count();
    }
    return summarize(times);
}

int itersForSize(int n) {
    if (n <= 512)  return 500;
    if (n <= 2048) return 250;
    if (n <= 4096) return 100;
    return 50;
}

static const std::string kCsvHeader = "image,filter,device,size,iters,mean_ms,median_ms,stddev_ms";

// Where the average of every numbered run is written.
static const std::string kSummaryPath = "summary_results.csv";

// One measured test: one row of a results file.
struct Result {
    std::string image;
    const char* filter;
    const char* device;
    int size;
    int iters;
    Stats stats;
};

void writeRow(std::ofstream& out, const Result& r) {
    out << r.image << "," << r.filter << "," << r.device << "," << r.size << ","
        << r.iters << "," << r.stats.mean << "," << r.stats.median << ","
        << r.stats.stddev << "\n";
}

// Writes the header and every result to path.
bool writeResults(const std::string& path, const std::vector<Result>& results) {
    std::ofstream out(path);
    if (!out) {
        std::cerr << "failed to open " << path << " for writing\n";
        return false;
    }
    out << kCsvHeader << "\n";
    for (const Result& r : results) writeRow(out, r);
    return true;
}

// Collects every regular file in dir, sorted by name so runs are reproducible.
// Files OpenCV cannot decode are skipped later, when they are loaded.
std::vector<std::string> listImageFiles(const std::string& dir) {
    std::vector<std::string> files;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (entry.is_regular_file()) files.push_back(entry.path().string());
    }
    std::sort(files.begin(), files.end());
    return files;
}

// Runs the full size sweep for one source image, adding one result per
// (filter, device, size). The image column holds the file name only.
void runImage(const std::string& path, const cv::Mat& base, bool haveGpu,
              std::vector<Result>& results) {
    std::string image = std::filesystem::path(path).filename().string();

    for (int n : kSizes) {
        // Setup is NOT timed: resize the photo and allocate all buffers once.
        Buffers b = makeBuffers(base, n);
        int iters = itersForSize(n);

        /* cpu tests start here */
        for (int i = 0; i < kCpuTestCount; ++i) {
            const Test& t = kCpuTests[i];
            Stats st = benchmark(t.fn, b, iters);
            results.push_back({image, t.name, "cpu", n, iters, st});
        }

        /* gpu tests start here */
        if (!haveGpu) continue;

        // Input is uploaded once outside the timed loop, so rows are kernel-only.
        GpuBuffers g = makeGpuBuffers(b.src);
        for (int i = 0; i < kGpuTestCount; ++i) {
            const GpuTest& t = kGpuTests[i];
            Stats st = benchmark(t.fn, g, iters);
            results.push_back({image, t.name, "gpu", n, iters, st});
        }
        freeGpuBuffers(g);
    }
}

// Returns true if a CUDA device is present and prints its name.
bool detectGpu() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count == 0) {
        std::cout << "no CUDA device found, running CPU tests only\n";
        return false;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name
              << ", compute capability " << prop.major << "." << prop.minor << "\n";
    return true;
}


// Parses the optional run count from the command line. Returns 0 if no
// count was given, or -1 if the argument is not a positive whole number.
int parseRunCount(int argc, char** argv) {
    if (argc < 2) return 0;

    std::string arg = argv[1];
    if (arg.empty() || arg.find_first_not_of("0123456789") != std::string::npos) {
        return -1;
    }
    int runs = std::stoi(arg);
    return (runs > 0) ? runs : -1;
}

// Loads every readable image in kImageDir, keeping each path alongside it.
void loadImages(const std::vector<std::string>& files,
                std::vector<std::string>& paths, std::vector<cv::Mat>& images) {
    for (const std::string& path : files) {
        cv::Mat base = cv::imread(path, cv::IMREAD_GRAYSCALE);
        if (base.empty()) {
            std::cerr << "skipping unreadable image: " << path << "\n";
            continue;
        }
        paths.push_back(path);
        images.push_back(base);
    }
}

// Runs the full sweep over every image once and returns every result.
std::vector<Result> runSweep(const std::vector<std::string>& paths,
                             const std::vector<cv::Mat>& images, bool haveGpu) {
    std::vector<Result> results;
    for (size_t i = 0; i < images.size(); ++i) {
        runImage(paths[i], images[i], haveGpu, results);
    }
    return results;
}

// Averages the mean, median and standard deviation of each test across runs.
// Every run tests the same images, filters and sizes in the same order, so
// row i of one run is the same test as row i of every other run.
std::vector<Result> averageRuns(const std::vector<std::vector<Result>>& runs) {
    std::vector<Result> summary = runs[0];
    for (size_t i = 0; i < summary.size(); ++i) {
        double mean = 0.0, median = 0.0, stddev = 0.0;
        for (const std::vector<Result>& run : runs) {
            mean   += run[i].stats.mean;
            median += run[i].stats.median;
            stddev += run[i].stats.stddev;
        }
        summary[i].stats.mean   = mean / runs.size();
        summary[i].stats.median = median / runs.size();
        summary[i].stats.stddev = stddev / runs.size();
    }
    return summary;
}

// Usage: ./main_test [runs]
//   no argument  one sweep, written to results.csv
//   runs = N     N sweeps, written to results1.csv ... resultsN.csv, plus
//                their average in summary_results.csv
int main(int argc, char** argv) {
    int runs = parseRunCount(argc, argv);
    if (runs < 0) {
        std::cerr << "usage: " << argv[0] << " [runs]  (runs must be a positive integer)\n";
        return 1;
    }

    std::vector<std::string> files = listImageFiles(kImageDir);
    if (files.empty()) {
        std::cerr << "no files found in " << kImageDir << "\n";
        return 1;
    }

    std::vector<std::string> paths;
    std::vector<cv::Mat> images;
    loadImages(files, paths, images);
    if (images.empty()) {
        std::cerr << "no readable images in " << kImageDir << "\n";
        return 1;
    }

    bool haveGpu = detectGpu();

    if (runs == 0) {
        std::cout << "starting test set 1 of 1" << std::endl;
        if (!writeResults(kResultsPath, runSweep(paths, images, haveGpu))) return 1;
        std::cout << "finished test set 1 of 1 -> " << kResultsPath << std::endl;
        return 0;
    }

    std::vector<std::vector<Result>> allRuns;
    for (int r = 1; r <= runs; ++r) {
        std::cout << "starting test set " << r << " of " << runs << std::endl;
        allRuns.push_back(runSweep(paths, images, haveGpu));

        std::string outPath = "results" + std::to_string(r) + ".csv";
        if (!writeResults(outPath, allRuns.back())) return 1;
        std::cout << "finished test set " << r << " of " << runs
                  << " -> " << outPath << std::endl;
    }

    if (!writeResults(kSummaryPath, averageRuns(allRuns))) return 1;
    std::cout << "all " << runs << " test sets done, averages written to "
              << kSummaryPath << std::endl;
    return 0;
}
