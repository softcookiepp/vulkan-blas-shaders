#ifndef CONSTANTS_H
#define CONSTANTS_H

#ifndef FLOAT_T
	#ifdef USE_COMPLEX
		// complex numbers are treated as 2-element vectors
		#define ELEM_T float
		#define FLOAT_T vec2
		#define FVEC2_T vec4
	#else
		#define FLOAT_T float
		#define FVEC2_T vec2
	#endif
#endif

#ifndef INT_T
#define INT_T int
#endif

#ifndef UINT_T
#define UINT_T uint
#endif

#ifndef IDX_T
#define IDX_T int
#endif

#ifndef UIDX_T
#define UIDX_T uint
#endif

#define INFINITY FLOAT_T(uintBitsToFloat(0x7F800000))

// memory orders
#define ROW_MAJOR 101
#define COLUMN_MAJOR 102

// transpose types
#define NO_TRANSPOSE 111
#define TRANSPOSE 112
#define CONJ_TRANSPOSE 113
#define CONJ_NO_TRANSPOSE 114

#endif
