/* Syntax-check the __global__ kernels without nvcc. Not a functional test. */
#define POM_NO_DRIVER 1
#define POM_CUDA_SHIM 1
#include "cuda_shim.h"
#include "search_cuda.cu"
int main() { return 0; }
