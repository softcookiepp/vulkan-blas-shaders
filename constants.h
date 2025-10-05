#ifndef CONSTANTS_H
#define CONSTANTS_H

#ifndef FLOAT_T
#define FLOAT_T float
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

#endif
