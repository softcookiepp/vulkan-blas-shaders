#version 450
#include "helpers.h"

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer a_buf { float a_arr[]; };
layout(set = 0, binding = 1) buffer b_buf { float b_arr[]; };
layout(set = 0, binding = 2) buffer c_buf { float c_arr[]; };
layout(set = 0, binding = 3) buffer s_buf { float s_arr[]; };

void main()
{
	int gid = int(gl_GlobalInvocationID.x);
	
	float a = a_arr[gid];
	float b = b_arr[gid];
	float r = sqrt(a*a + b*b);
	float z = 1.0 / r;
	float c = a * z;
	float s = b * z;
	c_arr[gid] = c;
	s_arr[gid] = s;
	// NOT FINISHED
}
