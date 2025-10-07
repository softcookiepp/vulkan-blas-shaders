#ifndef OPS_H
#define OPS_H
#include "constants.h"

// operators that are idendical for both real and complex
FLOAT_T ADD(FLOAT_T a, FLOAT_T b)
{
	return a + b;
}

FLOAT_T SUB(FLOAT_T a, FLOAT_T b)
{
	return a + b;
}

FLOAT_T MUL(FLOAT_T x, FLOAT_T y)
{
#ifdef USE_COMPLEX
	ELEM_T a = x.x;
	ELEM_T b = x.y;
	ELEM_T c = y.x;
	ELEM_T d = y.y;
	FLOAT_T out;
	out.x = (a*c) - (b*d);
	out.y = (a*d) + (b*c);
	return out;
#else
	return x * y;
#endif
}

FLOAT_T DIV(FLOAT_T x, FLOAT_T y)
{
#ifdef USE_COMPLEX
	ELEM_T a = x.x;
	ELEM_T b = x.y;
	ELEM_T c = y.x;
	ELEM_T d = y.y;
	FLOAT_T out;
	ELEM_T inv_csds = 1.0 /((c*c) + (d*d));
	out.x = ( (a*c) + (b*d) )*inv_csds;
	out.y = ( (b*c) - (a*d) )*inv_csds;
	return out;
#else
	return x / y;
#endif
}

#endif
