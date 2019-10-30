#pragma once
#include "cuda_runtime.h"

// Calculate the fast approximate cosine of the input argument.
__device__ float __cosf (float x);

// Calculate the fast approximate base 10 exponential of the input argument.
__device__ float __exp10f (float x);

// Calculate the fast approximate base e exponential of the input argument.
__device__ float __expf (float x);

// Add two floating point values in round-down mode.
__device__ float __fadd_rd (float x, float y);

// Add two floating point values in round-to-nearest-even mode.
__device__ float __fadd_rn (float x, float y);

// Add two floating point values in round-up mode.
__device__ float __fadd_ru (float x, float y);

// Add two floating point values in round-towards-zero mode.
__device__ float __fadd_rz (float x, float y);

// Divide two floating point values in round-down mode.
__device__ float __fdiv_rd (float x, float y);

// Divide two floating point values in round-to-nearest-even mode.
__device__ float __fdiv_rn (float x, float y);

// Divide two floating point values in round-up mode.
__device__ float __fdiv_ru (float x, float y);

// Divide two floating point values in round-towards-zero mode.
__device__ float __fdiv_rz (float x, float y);

// Calculate the fast approximate division of the input arguments.
__device__ float __fdividef (float x, float y);

// Compute x × y + z as a single operation, in round-down mode.
__device__ float __fmaf_rd (float x, float y, float z);

// Compute x × y + z as a single operation, in round-to-nearest-even mode.
__device__ float __fmaf_rn (float x, float y, float z);

// Compute x × y + z as a single operation, in round-up mode.
__device__ float __fmaf_ru (float x, float y, float z);

// Compute x × y + z as a single operation, in round-towards-zero mode.
__device__ float __fmaf_rz (float x, float y, float z);

// Multiply two floating point values in round-down mode.
__device__ float __fmul_rd (float x, float y);

// Multiply two floating point values in round-to-nearest-even mode.
__device__ float __fmul_rn (float x, float y);

// Multiply two floating point values in round-up mode.
__device__ float __fmul_ru (float x, float y);

// Multiply two floating point values in round-towards-zero mode.
__device__ float __fmul_rz (float x, float y);

// Compute 1/x in round-down mode.
__device__ float __frcp_rd (float x);

// Compute 1/x in round-to-nearest-even mode.
__device__ float __frcp_rn (float x);

// Compute 1/x in round-up mode.
__device__ float __frcp_ru (float x);

// Compute 1/x in round-towards-zero mode.
__device__ float __frcp_rz (float x);

// Compute x^(-1/2) in round-to-nearest-even mode.
__device__ float __frsqrt_rn (float x);

// Compute x^(1/2) in round-down mode.
__device__ float __fsqrt_rd (float x);

// Compute x^(1/2) in round-to-nearest-even mode.
__device__ float __fsqrt_rn (float x);

// Compute x^(1/2) in round-up mode.
__device__ float __fsqrt_ru (float x);

// Compute x^(1/2) in round-towards-zero mode.
__device__ float __fsqrt_rz (float x);

// Subtract two floating point values in round-down mode.
__device__ float __fsub_rd (float x, float y);

// Subtract two floating point values in round-to-nearest-even mode.
__device__ float __fsub_rn (float x, float y);

// Subtract two floating point values in round-up mode.
__device__ float __fsub_ru (float x, float y);

// Subtract two floating point values in round-towards-zero mode.
__device__ float __fsub_rz (float x, float y);

// Calculate the fast approximate base 10 logarithm of the input argument.
__device__ float __log10f (float x);

// Calculate the fast approximate base 2 logarithm of the input argument.
__device__ float __log2f (float x);

// Calculate the fast approximate base e logarithm of the input argument.
__device__ float __logf (float x);

// Calculate the fast approximate of x y .
__device__ float __powf (float x, float y);

// Clamp the input argument to [+0.0, 1.0].
__device__ float __saturatef (float x);

// Calculate the fast approximate of sine and cosine of the first input argument.
__device__ void __sincosf (float x, float* sptr, float* cptr);

// Calculate the fast approximate sine of the input argument.
__device__ float __sinf (float x);

// Calculate the fast approximate tangent of the input argument.
__device__ float __tanf (float x);

