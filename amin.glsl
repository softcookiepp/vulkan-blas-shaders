#version 450

#include "helpers.h"
#include "constants.h"

#define LX 512

layout(local_size_x = LX, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer inp_buffer { float inp[]; };
layout(set = 0, binding = 1) buffer outp_buffer { uint outp[]; };

layout(push_constant) uniform push
{
	int size;
	int inp_stride;
} consts;

shared float shared_mem[LX];
shared uint indices[LX];

void main()
{
	// initialize
	shared_mem[gl_LocalInvocationID.x] = 0.0;
	barrier();
	
	// loooop
	float to_shared = INFINITY;
	uint idx = -1;
	uint num_blocks = 1 + (consts.size / LX);
	uint lx = gl_LocalInvocationID.x;
	for (uint block = 0; block < num_blocks; block += 1)
	{
		uint pos = (LX*block) + lx;
		if (pos < consts.size)
		{
			uint elem = pos;
			pos = compute_index(elem, consts.size, consts.inp_stride);
			float val = abs(inp[pos]);
			if (val < to_shared)
			{
				to_shared = val;
				idx = elem;
			}
		}
	}
	shared_mem[lx] = to_shared;
	indices[lx] = idx;
	barrier();
	
	if (lx == 0)
	{
		float min_val = INFINITY;
		uint out_idx = 0;
		for (uint i = 0; i < LX; i += 1)
		{
			if (shared_mem[i] < min_val)
			{
				min_val = shared_mem[i];
				out_idx = indices[i];
			}
			else if(shared_mem[i] == min_val)
			{
				if(indices[i] < out_idx) out_idx = indices[i];
			}
		}
		outp[0] = out_idx;
	}
}
