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

// ze buffers
layout(binding = 0) buffer agm_buf { real agm[]; };
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
	//int a_rotated; // glslc will discard this since it isn't used
	// const __global realVFR* restrict agm,
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

// Local memory to store a tile of the matrix (for coalescing)
shared real tile[WPT3][WGS3];

// Local memory for the vector X
shared real xlm[WPT3];

// =================================================================================================

// Faster version of the kernel, assuming that:
// --> 'm' and 'n' are multiples of WGS3
// --> 'a_offset' is 0
// --> 'a_ld' is a multiple of VW3
// --> 'a_rotated' is 1
// --> 'do_conjugate' is 0
void main() {
  const real alpha = GetRealArg(consts.arg_alpha);
  const real beta = GetRealArg(consts.arg_beta);

  const int lid = int(gl_LocalInvocationID.x);
  const int lid_mod = lid % (WPT3/VW3);
  const int lid_div = lid / (WPT3/VW3);

  // Initializes the accumulation register
  real acc3;
  SetToZero(acc3);

  // Loops over tile-sized portions of the work
  for (int kwg=0; kwg < consts.n; kwg+=WPT3) {

    // Loads the vector X into local memory
    if (lid < WPT3) {
      xlm[lid] = xgm[(kwg + lid) * consts.x_inc + consts.x_offset];
    }

    // Loads the matrix A into local memory
    //#pragma unroll
    for (int _kl = 0; _kl < WPT3/VW3; _kl += 1) {
      const int x = (kwg/VW3) + lid_mod;
      const int y = int(gl_LocalInvocationID.x) * WGS3 + lid_div * (WPT3/VW3) + _kl;
      realVFR avec = agm[(consts.a_ld/VW3) * y + x];
      #if VW3 == 1
        tile[_kl*VW3 + 0][lid] = avec;
      #elif VW3 == 2
        tile[_kl*VW3 + 0][lid] = avec.x;
        tile[_kl*VW3 + 1][lid] = avec.y;
      #elif VW3 == 4
        tile[_kl*VW3 + 0][lid] = avec.x;
        tile[_kl*VW3 + 1][lid] = avec.y;
        tile[_kl*VW3 + 2][lid] = avec.z;
        tile[_kl*VW3 + 3][lid] = avec.w;
      #elif VW3 == 8
        tile[_kl*VW3 + 0][lid] = avec.s0;
        tile[_kl*VW3 + 1][lid] = avec.s1;
        tile[_kl*VW3 + 2][lid] = avec.s2;
        tile[_kl*VW3 + 3][lid] = avec.s3;
        tile[_kl*VW3 + 4][lid] = avec.s4;
        tile[_kl*VW3 + 5][lid] = avec.s5;
        tile[_kl*VW3 + 6][lid] = avec.s6;
        tile[_kl*VW3 + 7][lid] = avec.s7;
      #elif VW3 == 16
        tile[_kl*VW3 + 0][lid] = avec.s0;
        tile[_kl*VW3 + 1][lid] = avec.s1;
        tile[_kl*VW3 + 2][lid] = avec.s2;
        tile[_kl*VW3 + 3][lid] = avec.s3;
        tile[_kl*VW3 + 4][lid] = avec.s4;
        tile[_kl*VW3 + 5][lid] = avec.s5;
        tile[_kl*VW3 + 6][lid] = avec.s6;
        tile[_kl*VW3 + 7][lid] = avec.s7;
        tile[_kl*VW3 + 8][lid] = avec.s8;
        tile[_kl*VW3 + 9][lid] = avec.s9;
        tile[_kl*VW3 + 10][lid] = avec.sA;
        tile[_kl*VW3 + 11][lid] = avec.sB;
        tile[_kl*VW3 + 12][lid] = avec.sC;
        tile[_kl*VW3 + 13][lid] = avec.sD;
        tile[_kl*VW3 + 14][lid] = avec.sE;
        tile[_kl*VW3 + 15][lid] = avec.sF;
      #endif
    }

    // Synchronizes all threads in a workgroup
    barrier();

    // The multiply-add function (rotated)
    //#pragma unroll
    for (int _kl = 0; _kl < WPT3/VW3; _kl += 1) {
      //#pragma unroll
      for (int _v = 0; _v < VW3; _v += 1) {
        real aval = tile[lid_mod*VW3 + _v][lid_div * (WPT3/VW3) + _kl];
        real xval = xlm[_kl*VW3 + _v];
        MultiplyAdd(acc3, xval, aval);
      }
    }

    // Synchronizes all threads in a workgroup
    barrier();
  }

  // Stores the final result
  const int gid = int(gl_GlobalInvocationID.x);
  real yval = ygm[gid * consts.y_inc + consts.y_offset];
  AXPBY(ygm[gid * consts.y_inc + consts.y_offset], alpha, acc3, beta, yval);
}
