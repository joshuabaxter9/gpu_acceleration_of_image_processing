// compare.cu reads every results file written by main_test and compares the
// runtimes of each pair of images. Every image is resized to the same N x N
// before a test, so only rows with the same size are compared.
//
// Each results file is one run. For a pair of images (a, b), the signed
// difference b - a is computed inside each run, so both times come from the
// same conditions, and those per-run differences are then summarized across
// all runs. The per-run time used is median_ms, the least noisy column.
//
// One row is written to comparison.csv per (image pair, filter, device, size):
//   image_a         the baseline image
//   image_b         the image compared against it
//   filter          the algorithm
//   device          cpu or gpu
//   size            the N x N size both images were tested at
//   runs            how many runs had both images
//   mean_diff_ms    mean of (b - a) across runs; negative means b is faster
//   median_diff_ms  median of (b - a) across runs
//   mean_pct_diff   mean of 100 * (b - a) / a across runs; negative means b is
//                   faster
//
// Usage: ./compare_test [results files...]
//   no arguments  reads results.csv and results<N>.csv from the current
//                 directory, whichever exist
//   files given   reads exactly those files
// Output always goes to comparison.csv. The average of every column across
// all the files read is also written to summary_results.csv.

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

static const std::string kOutputPath  = "comparison.csv";
static const std::string kSummaryPath = "summary_results.csv";

static const std::string kResultsHeader =
    "image,filter,device,size,iters,mean_ms,median_ms,stddev_ms";

// One line of a results file:
//   image,filter,device,size,iters,mean_ms,median_ms,stddev_ms
struct Row {
    std::string image;
    std::string filter;
    std::string device;
    int size;
    int iters;
    double meanMs;
    double ms;  // median_ms, the value the comparison uses
    double stddevMs;
};

// Every measurement from one results file, keyed by measurementKey().
typedef std::map<std::string, double> Run;

// Splits one CSV line on commas. Fields never contain commas or quotes here.
std::vector<std::string> splitCsv(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string field;
    while (std::getline(ss, field, ',')) fields.push_back(field);
    return fields;
}

// Reads every data row of one results file. Returns false if the file is
// unreadable or its header is not the one main_test writes.
bool readResults(const std::string& path, std::vector<Row>& rows) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "failed to open " << path << "\n";
        return false;
    }

    std::string line;
    if (!std::getline(in, line) || line != kResultsHeader) {
        std::cerr << path << " does not have the expected header\n";
        return false;
    }

    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::vector<std::string> f = splitCsv(line);
        if (f.size() != 8) {
            std::cerr << "skipping malformed line in " << path << ": " << line << "\n";
            continue;
        }
        Row r;
        r.image  = f[0];
        r.filter = f[1];
        r.device = f[2];
        r.size     = std::stoi(f[3]);
        r.iters    = std::stoi(f[4]);
        r.meanMs   = std::stod(f[5]);
        r.ms       = std::stod(f[6]);
        r.stddevMs = std::stod(f[7]);
        if (r.ms <= 0.0) {
            std::cerr << "skipping row with non-positive runtime in " << path
                      << ": " << line << "\n";
            continue;
        }
        rows.push_back(r);
    }
    return true;
}

// Run number in a results file name: results.csv is 0, results7.csv is 7.
// Returns -1 for any other name.
int runNumber(const std::string& name) {
    static const std::regex pattern("results([0-9]*)\\.csv");
    std::smatch m;
    if (!std::regex_match(name, m, pattern)) return -1;
    return m[1].str().empty() ? 0 : std::stoi(m[1].str());
}

bool byRunNumber(const std::string& a, const std::string& b) {
    return runNumber(a) < runNumber(b);
}

// Finds results.csv and results<N>.csv in the current directory, in run order.
std::vector<std::string> findResultFiles() {
    std::vector<std::string> files;
    for (const auto& entry : std::filesystem::directory_iterator(".")) {
        if (!entry.is_regular_file()) continue;
        std::string name = entry.path().filename().string();
        if (runNumber(name) >= 0) files.push_back(name);
    }
    std::sort(files.begin(), files.end(), byRunNumber);
    return files;
}

// A filter/device pair, e.g. ("box3x3", "cpu").
struct Test {
    std::string filter;
    std::string device;
};

// Key for looking up one measurement: image + filter + device + size.
std::string measurementKey(const std::string& image, const std::string& filter,
                           const std::string& device, int size) {
    return image + "|" + filter + "|" + device + "|" + std::to_string(size);
}

// Adds the (filter, device) pairs of rows that are not already in tests, in
// the order they first appear, which is the order main_test ran them.
void addUniqueTests(const std::vector<Row>& rows, std::vector<Test>& tests,
                    std::set<std::string>& seen) {
    for (const Row& r : rows) {
        std::string key = r.filter + "|" + r.device;
        if (seen.count(key)) continue;
        seen.insert(key);
        tests.push_back({r.filter, r.device});
    }
}

double mean(const std::vector<double>& values) {
    double sum = 0.0;
    for (double v : values) sum += v;
    return sum / values.size();
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    size_t n = values.size();
    if (n % 2 == 1) return values[n / 2];
    return (values[n / 2 - 1] + values[n / 2]) / 2.0;
}

static const std::string kCsvHeader =
    "image_a,image_b,filter,device,size,runs,mean_diff_ms,median_diff_ms,mean_pct_diff";

void writeRow(std::ofstream& out, const std::string& a, const std::string& b,
              const Test& t, int size, int runs, double meanDiff,
              double medianDiff, double meanPct) {
    std::ostringstream line;
    line << a << "," << b << "," << t.filter << "," << t.device << "," << size
         << "," << runs << "," << meanDiff << "," << medianDiff << "," << meanPct;
    out       << line.str() << "\n";
    std::cout << line.str() << "\n";
}

// Compares image b against image a at one (filter, device, size) across all
// runs. Writes nothing if no run measured both images.
void comparePair(const std::string& a, const std::string& b, const Test& t,
                 int size, const std::vector<Run>& runs, std::ofstream& out) {
    std::string keyA = measurementKey(a, t.filter, t.device, size);
    std::string keyB = measurementKey(b, t.filter, t.device, size);

    std::vector<double> diffs;
    std::vector<double> pcts;
    for (const Run& run : runs) {
        Run::const_iterator itA = run.find(keyA);
        Run::const_iterator itB = run.find(keyB);
        if (itA == run.end() || itB == run.end()) continue;

        double diff = itB->second - itA->second;
        diffs.push_back(diff);
        pcts.push_back(100.0 * diff / itA->second);
    }
    if (diffs.empty()) return;

    writeRow(out, a, b, t, size, (int)diffs.size(), mean(diffs), median(diffs),
             mean(pcts));
}

// Running totals for one (image, filter, device, size) across all files.
struct SummaryRow {
    Row first;  // labels and iters, taken from the first file that has it
    double meanSum;
    double medianSum;
    double stddevSum;
    int count;
};

// Adds one file's rows to the totals. order keeps the keys in the order they
// were first seen, so the summary lists tests in the order main_test ran them.
void addToSummary(const std::vector<Row>& rows,
                  std::map<std::string, SummaryRow>& totals,
                  std::vector<std::string>& order) {
    for (const Row& r : rows) {
        std::string key = measurementKey(r.image, r.filter, r.device, r.size);
        std::map<std::string, SummaryRow>::iterator it = totals.find(key);
        if (it == totals.end()) {
            totals[key] = {r, r.meanMs, r.ms, r.stddevMs, 1};
            order.push_back(key);
            continue;
        }
        it->second.meanSum   += r.meanMs;
        it->second.medianSum += r.ms;
        it->second.stddevSum += r.stddevMs;
        it->second.count     += 1;
    }
}

// Writes the averages in the same format as a results file.
bool writeSummary(const std::map<std::string, SummaryRow>& totals,
                  const std::vector<std::string>& order) {
    std::ofstream out(kSummaryPath);
    if (!out) {
        std::cerr << "failed to open " << kSummaryPath << " for writing\n";
        return false;
    }
    out << kResultsHeader << "\n";
    for (const std::string& key : order) {
        const SummaryRow& s = totals.at(key);
        out << s.first.image << "," << s.first.filter << "," << s.first.device
            << "," << s.first.size << "," << s.first.iters << ","
            << s.meanSum / s.count << "," << s.medianSum / s.count << ","
            << s.stddevSum / s.count << "\n";
    }
    return true;
}

int main(int argc, char** argv) {
    std::vector<std::string> files;
    for (int i = 1; i < argc; ++i) files.push_back(argv[i]);
    if (files.empty()) files = findResultFiles();
    if (files.empty()) {
        std::cerr << "no results.csv or results<N>.csv files found; run "
                     "\"make run\" first\n";
        return 1;
    }

    // Index every measurement per run, and collect the distinct tests,
    // images and sizes across all runs.
    std::vector<Run> runs;
    std::vector<Test> tests;
    std::set<std::string> seenTests;
    std::set<std::string> imageSet;
    std::set<int> sizes;
    std::map<std::string, SummaryRow> totals;
    std::vector<std::string> summaryOrder;
    for (const std::string& path : files) {
        std::vector<Row> rows;
        if (!readResults(path, rows)) return 1;
        addToSummary(rows, totals, summaryOrder);

        Run run;
        for (const Row& r : rows) {
            run[measurementKey(r.image, r.filter, r.device, r.size)] = r.ms;
            imageSet.insert(r.image);
            sizes.insert(r.size);
        }
        addUniqueTests(rows, tests, seenTests);
        runs.push_back(run);
    }

    if (!writeSummary(totals, summaryOrder)) return 1;

    if (imageSet.size() < 2) {
        std::cerr << "need at least two images to compare, found "
                  << imageSet.size() << "\n";
        return 1;
    }

    std::ofstream out(kOutputPath);
    if (!out) {
        std::cerr << "failed to open " << kOutputPath << " for writing\n";
        return 1;
    }

    std::vector<std::string> images(imageSet.begin(), imageSet.end());

    std::cout << "comparing " << images.size() << " images across "
              << tests.size() << " filter/device tests, " << sizes.size()
              << " sizes and " << runs.size() << " run(s):\n";
    for (const std::string& path : files) std::cout << "  " << path << "\n";
    std::cout << "\n";

    out       << kCsvHeader << "\n";
    std::cout << kCsvHeader << "\n";

    for (const Test& t : tests) {
        for (int n : sizes) {
            for (size_t i = 0; i < images.size(); ++i) {
                for (size_t j = i + 1; j < images.size(); ++j) {
                    comparePair(images[i], images[j], t, n, runs, out);
                }
            }
        }
    }

    std::cout << "\ncomparison written to " << kOutputPath
              << ", averages written to " << kSummaryPath << "\n";
    return 0;
}
