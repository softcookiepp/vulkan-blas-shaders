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
	// so we are doing the whole thing in a single invocation.
	
	// store the entirety of x locally first
	FLOAT_T x_local[N];
	uint xidx_j = 0;
	for (uint j = 0; j < N; j += 1)
	{
		xidx_j = compute_index(j, N, consts.incx);
		x_local[j] = x[xidx_j];
	}
	barrier();
	
	// compute the solution
	uint Aidx;
	for (uint jr = N; jr > 0; jr -= 1)
	{
		uint j = jr - 1;
		Aidx = compute_mat_index(j, j, consts.lda, consts.transpose);
		FLOAT_T xlj = x_local[j] / A[Aidx];
		
		for (uint i = 0; i < j; i += 1)
		{
			Aidx = compute_mat_index(j, i, consts.lda, consts.transpose);
			x_local[i] = x_local[i] - A[Aidx]*xlj;
		}
		x_local[j] = xlj;
	}
	
	// write everything out
	for (uint j = 0; j < N; j += 1)
	{
		xidx_j = compute_index(j, N, consts.incx);
		x[xidx_j] = x_local[j];
	}
}
