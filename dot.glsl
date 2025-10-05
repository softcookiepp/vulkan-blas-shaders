#version 450
#include "constants.h"
#include "helpers.h"
#define LX 256

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer x_buffer { FLOAT_T x[]; };
layout(set = 0, binding = 1) buffer y_buffer { FLOAT_T y[]; };
layout(set = 0, binding = 2) buffer c_buffer { FLOAT_T c[]; };

layout(push_constant) uniform push
{
	uint size;
	int incx;
	int incy;
} consts;

shared FLOAT_T shared_mem[LX];

void main()
{
	// initialize
	shared_mem[gl_LocalInvocationID.x] = 0.0;
	barrier();
	
	// loooop
	FLOAT_T to_shared = 0.0;
	uint num_blocks = 1 + (consts.size / LX);
	uint lx = gl_LocalInvocationID.x;
	for (uint block = 0; block < num_blocks; block += 1)
	{
		uint pos = (LX*block) + lx;
		if (pos < consts.size)
		{
			// TODO: add offsets
			uint xidx = compute_index(pos, consts.size, consts.incx);
			uint yidx = compute_index(pos, consts.size, consts.incy);
			to_shared += (x[xidx] * y[yidx]);
		}
	}
	shared_mem[lx] = to_shared;
	barrier();
	if (lx == 0)
	{
		FLOAT_T outp_val = 0.0;
		for (uint i = 0; i < LX; i += 1)
		{
			outp_val += shared_mem[i];
		}
		c[0] = outp_val;
	}
}
