#version 450
#include "constants.h"
#include "helpers.h"

layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;
layout(constant_id = 0) const uint N = 1; // square matrix dimension, only applies to lower

layout(set = 0, binding = 0) buffer x_buf { FLOAT_T x[]; };
layout(set = 0, binding = 2) buffer A_buf { FLOAT_T A[]; };

layout(push_constant) uniform push
{
	uint order;
	uint transpose;
	uint n; // square matrix dimension
	FLOAT_T alpha;
	int incx;
	int incy;
	uint lda;
} consts;

shared FLOAT_T shared_mem[N];

void main()
{
	uint i = gl_WorkGroupID.x;
	uint j = gl_LocalInvocationID.x;
	
	uint xidx_j = compute_index()
	uint Aidx = compute_mat_index(i, j, consts.lda, consts.order);
	
}
