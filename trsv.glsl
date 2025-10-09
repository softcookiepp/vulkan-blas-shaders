#version 450
#include "constants.h"
#include "helpers.h"

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout(constant_id = 0) const uint N = 1; // square matrix dimension
layout(constant_id = 1) const bool LOWER = true; // whether to use lower or upper triangular
layout(constant_id = 2) const bool UNIT_DIAGONAL = false; // whether or not unit diagonals are used.

layout(set = 0, binding = 0) buffer x_buf { FLOAT_T x[]; };
layout(set = 0, binding = 1) buffer A_buf { FLOAT_T A[]; };

layout(push_constant) uniform push
{
	bool column_major;
	bool transpose;
	int incx;
	uint lda;
} consts;

void main()
{
	// these algorithms just aren't easily parallelizable, sadly.
	// so we are doing the whole thing in a single invocation, until I understand this stuff better.
	// store the entirety of x locally first
	FLOAT_T x_local[N];
	
	uint xidx = 0;
	uint Aidx;
	for (uint j = 0; j < N; j += 1)
	{
		xidx = compute_index(j, N, consts.incx);
		x_local[j] = x[xidx];
	}
#if 1
	FLOAT_T A_local[N][N];
	// load A into local memory
	for (uint i = 0; i < N; i += 1)
	{
		for (uint j = 0; j < N; j += 1)
		{
			
			if (consts.transpose)
				Aidx = compute_mat_index(i, j, consts.lda, consts.column_major);
			else
				Aidx = compute_mat_index(j, i, consts.lda, consts.column_major);
			A_local[j][i] = A[Aidx];
		}
	}
#endif
	
	if(LOWER)
	{
		// algorithm for lower triangular matrix
		// compute the solution
		for (uint i = 0; i < N; i += 1)
		{
			uint i2 = consts.transpose ? N - i - 1 : i;
			FLOAT_T x_i = x_local[i2];
			for (uint j = 0; j < i; j += 1)
			{
				uint j2 = consts.transpose ? N - j - 1 : j;
				Aidx = compute_mat_index(j2, i2, consts.lda, consts.column_major);
				x_i = x_i - A[Aidx]*x_local[j2];
				//x_i = x_i - A_local[j, i]*x_local[j];
			}
			Aidx = compute_mat_index(i2, i2, consts.lda, consts.column_major);
			x_local[i2] = x_i/A[Aidx];
		}
	}
	else
	{	
		// algorithm for upper triangular matrix
		// compute the solution
		if(!consts.transpose)
		{
			for (uint jr = N; jr > 0; jr -= 1)
			{
				uint j = jr - 1;
				Aidx = compute_mat_index(j, j, consts.lda, consts.column_major);
				FLOAT_T xlj = x_local[j] / A[Aidx];
				
				for (uint i = 0; i < j; i += 1)
				{
					Aidx = compute_mat_index(j, i, consts.lda, consts.column_major);
					x_local[i] = x_local[i] - A[Aidx]*xlj;
				}
				x_local[j] = xlj;
			}
		}
		else
		{
			for (uint jr = N; jr > 0; jr -= 1)
			{
				uint j = jr - 1;
				Aidx = compute_mat_index(j, j, consts.lda, consts.column_major);
				FLOAT_T xlj = x_local[j] / A_local[j][j];
				
				for (uint i = 0; i < j; i += 1)
				{
					//Aidx = compute_mat_index(j, i, consts.lda, consts.column_major);
					//x_local[i] = x_local[i] - A[Aidx]*xlj;
					x_local[i] = x_local[i] - A_local[j][i]*xlj;
				}
				x_local[j] = xlj;
			}
		}
	}
	// write everything out
	for (uint j = 0; j < N; j += 1)
	{
		xidx = compute_index(j, N, consts.incx);
		x[xidx] = x_local[j];
	}
}
