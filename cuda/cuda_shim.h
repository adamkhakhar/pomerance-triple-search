/* Minimal CUDA shim: lets a host C++ compiler syntax-check the kernels when
   no nvcc is available.  Never used in a real build. */
#pragma once
#include <cstdint>
#include <cstdlib>
#define __global__
#define __device__
#define __host__
struct pom_dim3 { unsigned x, y, z; };
static pom_dim3 blockIdx{0,0,0}, blockDim{1,1,1}, threadIdx{0,0,0}, gridDim{1,1,1};
typedef int cudaError_t;
static const cudaError_t cudaSuccess = 0;
static inline const char *cudaGetErrorString(cudaError_t) { return ""; }
template <class T> static inline unsigned long long atomicAdd(T *p, unsigned long long v)
{ unsigned long long o = (unsigned long long)*p; *p = (T)(o + v); return o; }
static inline int atomicCAS(int *p, int c, int v)
{ int o = *p; if (o == c) *p = v; return o; }
