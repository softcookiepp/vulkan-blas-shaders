#version 450
#include "../common.glsl"
#include "xtrsv-common.glsl"

layout(binding = 0) buffer A_buf { real A[]; };
layout(binding = 1) buffer b_buf { real b[]; };
layout(binding = 2) buffer x_buf { real x[]; };

layout(push_constant) uniform const_struct
{
	int n;
	// A
	int a_offset;
	int a_ld;
	// b
	int b_offset;
	int b_inc;
	// x
	int x_offset;
	int x_inc;
	int is_transposed;
	int is_unit_diagonal;
	int do_conjugate;
} consts;

shared real alm[TRSV_BLOCK_SIZE][TRSV_BLOCK_SIZE];
shared real xlm[TRSV_BLOCK_SIZE];

void main() {
  const int tid = int(gl_LocalInvocationID.x);

  // Pre-loads the data into local memory
  if (tid < consts.n) {
    Subtract(xlm[tid], b[tid*consts.b_inc + consts.b_offset], x[tid*consts.x_inc + consts.x_offset]);
    if (consts.is_transposed == 0) {
      for (int i = 0; i < consts.n; ++i) {
        alm[i][tid] = A[i + tid*consts.a_ld + consts.a_offset];
      }
    }
    else {
      for (int i = 0; i < consts.n; ++i) {
        alm[i][tid] = A[tid + i*consts.a_ld + consts.a_offset];
      }
    }
    if (consts.do_conjugate > 0) {
      for (int i = 0; i < consts.n; ++i) {
        COMPLEX_CONJUGATE(alm[i][tid]);
      }
    }
  }
  barrier();

  // Computes the result (single-threaded for now)
  if (tid == 0) {
    for (int i = 0; i < consts.n; ++i) {
      for (int j = 0; j < i; ++j) {
        MultiplySubtract(xlm[i], alm[i][j], xlm[j]);
      }
      if (consts.is_unit_diagonal == 0) { DivideFull(xlm[i], xlm[i], alm[i][i]); }
    }
  }
  barrier();

  // Stores the results
  if (tid < consts.n) {
    x[tid*consts.x_inc + consts.x_offset] = xlm[tid];
  }
}
