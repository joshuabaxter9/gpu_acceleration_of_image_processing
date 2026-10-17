# makefile will be used to efficiently and easily run the benchmarks

NVCC      := nvcc
# -arch=sm_86 targets the RTX 3070 Ti. Use -arch=native to build for whatever
# GPU is installed.
CXXFLAGS  := -std=c++17 -O2 -arch=sm_86 -diag-suppress 611
OPENCV    := $(shell pkg-config --cflags --libs opencv4)

TARGET    := main_test
COMPARE   := compare_test

SRCS      := main.cu \
             cpu_implementation/cpu_filters.cpp \
             gpu_implementation/gpu_filters.cu \
             gpu_implementation/box_blur.cu \
             gpu_implementation/2d_gaussian_blur.cu \
             gpu_implementation/median_filter.cu \
             gpu_implementation/sobel_filter.cu \
             gpu_implementation/unsharp_masking.cu
HDRS      := cpu_implementation/cpu_filters.h \
             gpu_implementation/gpu_filters.h

.PHONY: all run compare clean

all: $(TARGET) $(COMPARE)

$(COMPARE): compare.cu
	$(NVCC) $(CXXFLAGS) compare.cu -o $@

$(TARGET): $(SRCS) $(HDRS)
	$(NVCC) $(CXXFLAGS) $(SRCS) -o $@ $(OPENCV)

# "make run" does one sweep and writes results.csv.
# "make run 10" does 10 sweeps and writes results1.csv ... results10.csv,
# plus their average in summary_results.csv.
# make treats the 10 as a target, so it is read from MAKECMDGOALS and given
# an empty rule so make does not try to build it.
ifeq ($(firstword $(MAKECMDGOALS)),run)
  RUNS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  $(eval $(RUNS):;@:)
endif

run: $(TARGET)
	./$(TARGET) $(RUNS)

# Build if needed, then compare every results file -> comparison.csv, and
# average them all -> summary_results.csv
compare: $(COMPARE)
	./$(COMPARE)

clean:
	rm -f $(TARGET) $(COMPARE) results.csv results[0-9]*.csv summary_results.csv comparison.csv
