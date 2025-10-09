#version 450
#include "constants.h"
#include "helpers.h"

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer a_buf { FLOAT_T a_arr[]; };
layout(set = 0, binding = 1) buffer b_buf { FLOAT_T b_arr[]; };
layout(set = 0, binding = 2) buffer c_buf { FLOAT_T c_arr[]; };
layout(set = 0, binding = 3) buffer s_buf { FLOAT_T s_arr[]; };

void main()
{
	int gid = int(gl_GlobalInvocationID.x);
	
	FLOAT_T a = a_arr[gid];
	FLOAT_T b = b_arr[gid];
	FLOAT_T r = sqrt(a*a + b*b);
	FLOAT_T z = 1.0 / r;
	FLOAT_T c = a * z;
	FLOAT_T s = b * z;
	
	c_arr[gid] = c;
	s_arr[gid] = s;
	a_arr[gid] = r;
	b_arr[gid] = z;
}
