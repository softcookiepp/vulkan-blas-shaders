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
	FLOAT_T params[5];
} consts;

void main()
{
	int gid = int(gl_GlobalInvocationID.x);
	uint ix = compute_index(gid, consts.n, consts.incx);
	uint iy = compute_index(gid, consts.n, consts.incy);
	
	FLOAT_T xval = x[ix];
	FLOAT_T yval = y[iy];
	
	// H matrix should be column-major
	FLOAT_T flag = consts.params[0];
	FLOAT_T h11 = flag == -1.0 || flag == 1.0 ? consts.params[1] : 1.0;
	FLOAT_T h21 = flag != -2.0 ? (flag == -1.0 || flag == 0.0 ? consts.params[2]: 1.0) : 0.0;
	FLOAT_T h12 = flag != -2.0 ? (flag == -1.0 || flag == 0.0 ? consts.params[3]: -1.0) : 0.0;;
	FLOAT_T h22 = flag == -1.0 || flag == 1.0 ? consts.params[4] : 1.0;
	
	x[ix] = h11*xval + h12*yval;
	y[iy] = h21*xval + h22*yval;
	
	
}
