# CUDA build.  ARCH defaults to Ada (RTX 6000 Ada / 4090 = sm_89);
# use sm_80 for A100, sm_90 for H100, sm_86 for A6000/3090.
NVCC ?= nvcc
ARCH ?= sm_89
LANES ?= 4
NVCCFLAGS ?= -O3 -std=c++17 -arch=$(ARCH) -DPOM_COVER_LANES=$(LANES)

search_cuda: search_cuda.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# Host-side validation of the exact device math: no GPU required.
# Proves the transport identity, the cover walk, order-32 emission, the
# 2x hazard, and finds real triples that tools/vpp.py accepts.
host_test: host_test.cpp search_cuda.cu
	$(CXX) -O2 -std=c++17 -DPOM_HOST_TEST -o $@ host_test.cpp

# Syntax-check the __global__ kernels without nvcc (CI on a GPU-less box).
.PHONY: kernel-syntax
kernel-syntax: kernel_syntax.cpp cuda_shim.h search_cuda.cu
	$(CXX) -O0 -std=c++17 -fsyntax-only kernel_syntax.cpp
	@echo "kernel syntax: OK"

.PHONY: check
check: host_test kernel-syntax
	./host_test

.PHONY: clean
clean:
	rm -f search_cuda host_test
