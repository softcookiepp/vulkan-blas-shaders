#version 450
#include "constants.h"
#include "helpers.h"

layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;
layout(constant_id = 0) const uint N = 1; // square matrix dimension, only applies to lower

layout(set = 0, binding = 0) buffer x_buf { FLOAT_T x[]; };
layout(set = 0, binding = 1) buffer A_buf { FLOAT_T A[]; };

layout(push_constant) uniform push
{
	uint order;
	uint transpose;
	int incx;
	uint lda;
} consts;

shared FLOAT_T shared_mem[N];

void main()
{
	uint i = gl_WorkGroupID.x;
	uint j = gl_LocalInvocationID.x;
	
	uint xidx_j = compute_index(j, N, consts.incx);
	uint xidx_i = compute_index(i, N, consts.incx);
	uint Aidx = compute_mat_index(j, i, consts.lda, consts.order);
#if 0
	A[Aidx] = FLOAT_T(j);
#endif
	if (j < i)
		shared_mem[j] = MUL(A[Aidx], x[xidx_j]);
	barrier();
	if (j == 0)
	{
		FLOAT_T x_i = x[xidx_i];
		for (uint j_tmp = 0; j_tmp < N; j_tmp += 1)
		{
			x_i = x_i - shared_mem[j_tmp];
		}
		Aidx = compute_mat_index(i, i, consts.lda, consts.order);
		x[xidx_i] = x_i / A[Aidx];
	}
}
