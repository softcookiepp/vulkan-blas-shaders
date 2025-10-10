#version 450
#include "../common.glsl"

layout(local_size_x = 16) in;

layout(binding = 0) writeonly buffer dest_buf { real dest[]; };

layout(push_constant) uniform const_layout
{
	int n;
	int inc;
	int offset;
	real_arg arg_value; // the fill value, presumably
} consts;

void main()
{
	const real value = GetRealArg(consts.arg_value);
	const int tid = int(gl_GlobalInvocationID.x);
	if (tid < consts.n)
	{
		dest[tid*consts.inc + consts.offset] = value;
	}
}
