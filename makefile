# makefile will be used to efficiently and easily run the benchmarks

NVCC      := nvcc
CXXFLAGS  := -std=c++17 -O2 -diag-suppress 611
OPENCV    := $(shell pkg-config --cflags --libs opencv4)

TARGET    := main_test

# CPU-only build. When the GPU side is added, append the kernel files here,
# e.g. gpu_implementation/gpu_filters.cu gpu_implementation/box_blur.cu ...,
# add gpu_implementation/gpu_filters.h to HDRS, and add -arch=sm_86 (or
# -arch=native) to CXXFLAGS so the kernels are built for the target GPU.
SRCS      := main.cu \
             cpu_implementation/cpu_filters.cpp
HDRS      := cpu_implementation/cpu_filters.h

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRCS) $(HDRS)
	$(NVCC) $(CXXFLAGS) $(SRCS) -o $@ $(OPENCV)

# Build if needed, then run the full sweep and write results.csv
run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) results.csv
