#version 450
#include "constants.h"
#include "helpers.h"

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;


layout(set = 0, binding = 0) buffer x_buf { FLOAT_T x[]; };
layout(set = 0, binding = 1) buffer y_buf { FLOAT_T y[]; };
layout(set = 0, binding = 2) buffer A_buf { FLOAT_T A[]; };

layout(push_constant) uniform push
{
	uint order;
	uint transpose;
	uint m; // A column size, x size
	uint n; // A row size, y size
	FLOAT_T alpha;
	int incx;
	int incy;
	uint lda;
} consts;

void main()
{
	if (consts.order == ROW_MAJOR)
	{
		uint ypos = gl_WorkGroupID.x;
		uint xpos = gl_WorkGroupID.y;
		
		// y is internally transposed.
		uint xidx = compute_index(xpos, consts.m, consts.incx);
		uint yidx = compute_index(ypos, consts.n, consts.incy);
		
		uint column_elem = consts.transpose == NO_TRANSPOSE ? xpos : ypos;
		uint row_elem = consts.transpose == NO_TRANSPOSE ? ypos : xpos;
		uint Aidx = compute_mat_index(row_elem, column_elem, consts.lda, consts.order);
		A[Aidx] = MUL(consts.alpha, MUL(x[xidx], y[yidx]) ) + A[Aidx];
	}
	else
	{
		uint ypos = gl_WorkGroupID.x;
		uint xpos = gl_WorkGroupID.y;
		
		// y is internally transposed.
		uint xidx = compute_index(xpos, consts.m, consts.incx);
		uint yidx = compute_index(ypos, consts.n, consts.incy);
		
		uint column_elem = consts.transpose == NO_TRANSPOSE ? ypos : xpos;
		uint row_elem = consts.transpose == NO_TRANSPOSE ? xpos : ypos;
		uint Aidx = compute_mat_index(row_elem, column_elem, consts.lda, consts.order);
		A[Aidx] = MUL(consts.alpha, MUL(x[xidx], y[yidx]) ) + A[Aidx];
	}
}
