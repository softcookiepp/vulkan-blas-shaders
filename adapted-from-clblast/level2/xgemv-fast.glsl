#version 450
#include "../common.glsl"
// =================================================================================================
// This file is part of the CLBlast project. Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the Xgemv kernel (fast versions) for matrix-vector multiplication.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.

// =================================================================================================

// Parameters set by the tuner or by the database. Here they are given a basic default value in case
// this kernel file is used outside of the CLBlast library.

// 1: For the full version, see 'xgemv.opencl'

// 2: For the fast version
#ifndef WGS2
  #define WGS2 64     // The local work-group size
#endif
#ifndef WPT2
  #define WPT2 1      // The amount of work-per-thread
#endif
#ifndef VW2
  #define VW2 1       // Vector width of matrix A loads
#endif

// 3: For the fast rotated version
#ifndef WGS3
  #define WGS3 64     // The local work-group size
#endif
#ifndef WPT3
  #define WPT3 1      // The tile-size
#endif
#ifndef VW3
  #define VW3 1       // Vector width of matrix A loads
#endif

// =================================================================================================

// Data-widths for the 'fast' kernel
#if VW2 == 1
  #define realVF real
#elif VW2 == 2
  #define realVF real2
#elif VW2 == 4
  #define realVF real4
#elif VW2 == 8
  #define realVF real8
#elif VW2 == 16
  #define realVF real16
#endif

// Data-widths for the 'fast' kernel with rotated matrix
#if VW3 == 1
  #define realVFR real
#elif VW3 == 2
  #define realVFR real2
#elif VW3 == 4
  #define realVFR real4
#elif VW3 == 8
  #define realVFR real8
#elif VW3 == 16
  #define realVFR real16
#endif

// hon hon hon le buffers
layout(binding = 0) buffer agm_buf { realVF agm[]; };
layout(binding = 1) buffer xgm_buf { real xgm[]; };
layout(binding = 2) buffer ygm_buf { real ygm[]; };

// default local size
layout(local_size_x = WGS2, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform push
{
	int m;
	int n;
	real_arg arg_alpha;
	real_arg arg_beta;
	//int a_rotated;
	// const __global realVF* restrict agm,
	//int a_offset;
	int a_ld;
	// const __global real* restrict xgm,
	int x_offset;
	int x_inc;
	// __global real* ygm,
	int y_offset;
	int y_inc;
	//int do_conjugate;
	//int parameter;
	// int kl_unused;
	// int ku_unused;
} consts;

// Local memory for the vector X
shared real xlm[WGS2];

// =================================================================================================

// Faster version of the kernel, assuming that:
// --> 'm' and 'n' are multiples of WGS2
// --> 'a_offset' is 0
// --> 'a_ld' is a multiple of VW2
// --> 'a_rotated' is 0
// --> 'do_conjugate' is 0
void main() {
  const real alpha = GetRealArg(consts.arg_alpha);
  const real beta = GetRealArg(consts.arg_beta);

  // Initializes the accumulation registers
  // #pragma promote_to_registers
  real acc2[WPT2];
  // #pragma unroll
  for (int _w = 0; _w < WPT2; _w += 1) {
    SetToZero(acc2[_w]);
  }

  // Loops over work-group sized portions of the work
  for (int kwg=0; kwg<consts.n; kwg+=WGS2) {

    // Loads the vector X into local memory
    const int lid = int(gl_LocalInvocationID.x);
    xlm[lid] = xgm[(kwg + lid)*consts.x_inc + consts.x_offset];

    // Synchronizes all threads in a workgroup
    barrier();

    // The multiply-add function (not rotated)
    //#pragma unroll
    for (int _kl = 0; _kl < WGS2; _kl += 1) {
      const int k = kwg + _kl;
      //#pragma unroll
      for (int _w = 0; _w < WPT2/VW2; _w += 1) {
        const int gid = (WPT2/VW2)*int(gl_GlobalInvocationID.x) + _w;
        realVF avec = agm[(consts.a_ld/VW2)*k + gid];
        #if VW2 == 1
          MultiplyAdd(acc2[VW2*_w+0], xlm[_kl], avec);
        #elif VW2 == 2
          MultiplyAdd(acc2[VW2*_w+0], xlm[_kl], avec.x);
          MultiplyAdd(acc2[VW2*_w+1], xlm[_kl], avec.y);
        #elif VW2 == 4
          MultiplyAdd(acc2[VW2*_w+0], xlm[_kl], avec.x);
          MultiplyAdd(acc2[VW2*_w+1], xlm[_kl], avec.y);
          MultiplyAdd(acc2[VW2*_w+2], xlm[_kl], avec.z);
          MultiplyAdd(acc2[VW2*_w+3], xlm[_kl], avec.w);
        #elif VW2 == 8
          MultiplyAdd(acc2[VW2*_w+0], xlm[_kl], avec.s0);
          MultiplyAdd(acc2[VW2*_w+1], xlm[_kl], avec.s1);
          MultiplyAdd(acc2[VW2*_w+2], xlm[_kl], avec.s2);
          MultiplyAdd(acc2[VW2*_w+3], xlm[_kl], avec.s3);
          MultiplyAdd(acc2[VW2*_w+4], xlm[_kl], avec.s4);
          MultiplyAdd(acc2[VW2*_w+5], xlm[_kl], avec.s5);
          MultiplyAdd(acc2[VW2*_w+6], xlm[_kl], avec.s6);
          MultiplyAdd(acc2[VW2*_w+7], xlm[_kl], avec.s7);
        #elif VW2 == 16
          MultiplyAdd(acc2[VW2*_w+0], xlm[_kl], avec.s0);
          MultiplyAdd(acc2[VW2*_w+1], xlm[_kl], avec.s1);
          MultiplyAdd(acc2[VW2*_w+2], xlm[_kl], avec.s2);
          MultiplyAdd(acc2[VW2*_w+3], xlm[_kl], avec.s3);
          MultiplyAdd(acc2[VW2*_w+4], xlm[_kl], avec.s4);
          MultiplyAdd(acc2[VW2*_w+5], xlm[_kl], avec.s5);
          MultiplyAdd(acc2[VW2*_w+6], xlm[_kl], avec.s6);
          MultiplyAdd(acc2[VW2*_w+7], xlm[_kl], avec.s7);
          MultiplyAdd(acc2[VW2*_w+8], xlm[_kl], avec.s8);
          MultiplyAdd(acc2[VW2*_w+9], xlm[_kl], avec.s9);
          MultiplyAdd(acc2[VW2*_w+10], xlm[_kl], avec.sA);
          MultiplyAdd(acc2[VW2*_w+11], xlm[_kl], avec.sB);
          MultiplyAdd(acc2[VW2*_w+12], xlm[_kl], avec.sC);
          MultiplyAdd(acc2[VW2*_w+13], xlm[_kl], avec.sD);
          MultiplyAdd(acc2[VW2*_w+14], xlm[_kl], avec.sE);
          MultiplyAdd(acc2[VW2*_w+15], xlm[_kl], avec.sF);
        #endif
      }
    }

    // Synchronizes all threads in a workgroup
    barrier();
  }

  // Stores the final result
  // #pragma unroll
  for (int _w = 0; _w < WPT2; _w += 1) {
    const int gid = WPT2*int(gl_GlobalInvocationID.x) + _w;
    real yval = ygm[gid*consts.y_inc + consts.y_offset];
    AXPBY(ygm[gid*consts.y_inc + consts.y_offset], alpha, acc2[_w], beta, yval);
  }
}
