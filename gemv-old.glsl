#version 450
#include "constants.h"
#include "helpers.h"

layout(constant_id = 0) const uint LX = 1; // vector size; this MUST be specified at time of pipeline construction
layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer x_buf { FLOAT_T x[]; };
layout(set = 0, binding = 1) buffer A_buf { FLOAT_T A[]; };
layout(set = 0, binding = 2) buffer y_buf { FLOAT_T y[]; };

layout(push_constant) uniform push
{
	bool transpose;
	FLOAT_T a;
	uint lda;
	int incx;
	FLOAT_T b;
	int incy;
	uint m;
	uint n;
} consts;

shared FLOAT_T shared_mem[LX];

void main()
{
	uint row_elem = gl_WorkGroupID.x;
	uint column_elem = gl_LocalInvocationID.x;
	
	// each column of A will only be used once.
	// the entirety of x will be used for every computed value of y
	// which means, the partial sums of elements of x and A will be stored in shared memory.
	// TODO: if A is too large, the system may run out of shared memory. Take this into account...
	uint xidx = compute_index(column_elem, LX, consts.incx);
	
	// index of A will be computed by using the row and column position.
	// the column and row position will be dependent on the transpose status
	
	uint Aidx = compute_mat_index(row_elem, column_elem, consts.lda, consts.transpose);
	
	shared_mem[column_elem] = MUL(x[xidx], A[Aidx]);
	barrier();
	
	// from here on out, only the first thread will need to do anything
	if (column_elem == 0)
	{
		// sum all the partials
		FLOAT_T xval = 0.0;
		for (uint i = 0; i < LX; i += 1)
		{
			xval += shared_mem[i];
		}
		
		// compute the final result
		uint yidx = compute_index(row_elem, consts.m, consts.incy);
		FLOAT_T yval = y[yidx];
		y[yidx] = MUL(xval, consts.a) + MUL(yval, consts.b);
	}
}
