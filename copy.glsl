#version 450
#include "constants.h"
#include "helpers.h"

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer x_buf { FLOAT_T x[]; };
layout(set = 0, binding = 1) buffer y_buf { FLOAT_T y[]; };

layout(push_constant) uniform push
{
	uint n; // size 
	int incx;
	int incy;
} consts;

void main()
{
	int gid = int(gl_GlobalInvocationID.x);
	uint ix = compute_index(gid, consts.n, consts.incx);
	uint iy = compute_index(gid, consts.n, consts.incy);
	FLOAT_T xval = x[ix];
	y[iy] = xval;
}
