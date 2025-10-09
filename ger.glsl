#version 450
#include "constants.h"
#include "helpers.h"

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;


layout(set = 0, binding = 0) buffer x_buf { FLOAT_T x[]; };
layout(set = 0, binding = 1) buffer y_buf { FLOAT_T y[]; };
layout(set = 0, binding = 2) buffer A_buf { FLOAT_T A[]; };

layout(push_constant) uniform push
{
	bool transpose;
	uint m; // A column size, x size
	uint n; // A row size, y size
	FLOAT_T alpha;
	int incx;
	int incy;
	uint lda;
} consts;

void main()
{
	// column elem is the x vector position
	// row elem is the y vector position
	uint row_elem = gl_WorkGroupID.x;
	uint column_elem = gl_WorkGroupID.y;
	
	uint xidx = compute_index(column_elem, consts.m, consts.incx);
	uint yidx = compute_index(row_elem, consts.n, consts.incy);

	uint Aidx = compute_mat_index(row_elem, column_elem, consts.lda, consts.transpose);

	A[Aidx] = MUL(consts.alpha, MUL(x[xidx], y[yidx]) ) + A[Aidx];
}
