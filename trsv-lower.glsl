#version 450
#include "constants.h"
#include "helpers.h"

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout(constant_id = 0) const uint N = 1; // square matrix dimension, only applies to lower

layout(set = 0, binding = 0) buffer x_buf { FLOAT_T x[]; };
layout(set = 0, binding = 1) buffer A_buf { FLOAT_T A[]; };

layout(push_constant) uniform push
{
	bool transpose;
	int incx;
	uint lda;
} consts;

void main()
{
	// this algorithm just isn't very parallelizable, sadly.
	// so we are doing the whole thing in a single invocation, until I understand this stuff better.
	
	// store the entirety of x locally first
	FLOAT_T x_local[N];
	uint xidx_i = 0;
	for (uint i = 0; i < N; i += 1)
	{
		xidx_i = compute_index(i, N, consts.incx);
		x_local[i] = x[xidx_i];
	}
	
	// compute the solution
	uint Aidx;
	for (uint i = 0; i < N; i += 1)
	{
		compute_index(i, N, consts.incx);
		FLOAT_T x_i = x_local[i];
		
		for (uint j = 0; j < i; j += 1)
		{
			Aidx = compute_mat_index(j, i, consts.lda, consts.transpose);
			x_i = x_i - A[Aidx]*x_local[j];
		}
		Aidx = compute_mat_index(i, i, consts.lda, consts.transpose);
		x_local[i] = x_i/A[Aidx];
	}
	
	// write everything out
	for (uint i = 0; i < N; i += 1)
	{
		xidx_i = compute_index(i, N, consts.incx);
		x[xidx_i] = x_local[i];
	}
}
