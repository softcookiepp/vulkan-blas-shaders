#version 450
#include "helpers.h"

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer x_buf { float x[]; };
layout(set = 0, binding = 1) buffer y_buf { float y[]; };

layout(push_constant) uniform push
{
	uint n; // size
	int incx;
	int incy;
	float c;
	float s;
} consts;

void main()
{
	int gid = int(gl_GlobalInvocationID.x);
	uint ix = compute_index(gid, consts.n, consts.incx);
	uint iy = compute_index(gid, consts.n, consts.incy);
	
	float xval = x[ix];
	float yval = y[iy];
	
	x[ix] = (consts.c*xval) + (consts.s*yval);
	y[iy] = (consts.c*yval) - (consts.s*xval);
}
