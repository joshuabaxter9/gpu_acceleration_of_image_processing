# makefile will be used to efficiently and easily run the benchmarks

NVCC      := nvcc
# sm_86 is the RTX 3070 Ti. Change if you build on a different GPU.
CXXFLAGS  := -std=c++17 -O2 -arch=sm_86 -diag-suppress 611
OPENCV    := $(shell pkg-config --cflags --libs opencv4)

TARGET    := main_test
SRCS      := main.cu \
             cpu_implementation/cpu_filters.cpp \
             gpu_implementation/gpu_filters.cu \
             gpu_implementation/box_blur.cu \
             gpu_implementation/2d_gaussian_blur.cu \
             gpu_implementation/median_filter.cu \
             gpu_implementation/sobel_filter.cu \
             gpu_implementation/unsharp_masking.cu
HDRS      := cpu_implementation/cpu_filters.h \
             gpu_implementation/gpu_filters.h \
             gpu_implementation/gpu_common.cuh

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRCS) $(HDRS)
	$(NVCC) $(CXXFLAGS) $(SRCS) -o $@ $(OPENCV)

# Build if needed, then run the full sweep and write results.csv
run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) results.csv
