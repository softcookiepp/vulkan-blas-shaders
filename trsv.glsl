#version 450
#include "constants.h"
#include "helpers.h"

layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;
layout(constant_id = 0) const uint N = 1; // square matrix dimension, only applies to lower

layout(set = 0, binding = 0) buffer x_buf { FLOAT_T x[]; };
layout(set = 0, binding = 1) buffer y_buf { FLOAT_T y[]; };
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

void main()
{
	uint i = gl_WorkGroupID.x;
	uint j = gl_WorkGroupID.y;
	
	
}
