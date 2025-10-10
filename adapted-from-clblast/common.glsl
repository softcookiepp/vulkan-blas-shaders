#ifndef PRECISION
  #define PRECISION 32      // Data-types: half, single or double precision, complex or regular
#endif

// =================================================================================================

// Enable support for half-precision
#if PRECISION == 16
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#endif

// Enable support for double-precision
#if PRECISION == 64 || PRECISION == 6464
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#endif

// Half-precision
#if PRECISION == 16
  #define real float16_t
  #define real2 f16vec2
  #define real4 f16vec4
  #define real8 f16vec8
  #define real16 f16vec16
  #define ZERO 0
  #define ONE 1
  #define SMALLEST -1.0e14

// Single-precision
#elif PRECISION == 32
  #define real float
  #define real2 vec2
  #define real4 vec4
  #define real8 vec8
  #define real16 vec16
  #define ZERO 0.0f
  #define ONE 1.0f
  #define SMALLEST -1.0e37f

// Double-precision 
#elif PRECISION == 64
  #define real double
  #define real2 dvec2 
  #define real4 dvec4
  #define real8 dvec8
  #define real16 dvec16
  #define ZERO 0.0
  #define ONE 1.0
  #define SMALLEST -1.0e37

// Complex single-precision
#elif PRECISION == 3232
  #define real float2
  struct real2 {real x; real y;};
  struct real4 {real x; real y; real z; real w;};
  struct real8 {real s0; real s1; real s2; real s3;
                  real s4; real s5; real s6; real s7;};
  struct real16 {real s0; real s1; real s2; real s3;
                 real s4; real s5; real s6; real s7;
                 real s8; real s9; real sA; real sB;
                 real sC; real sD; real sE; real sF;};
  #define ZERO 0.0f
  #define ONE 1.0f
  #define SMALLEST -1.0e37f

// Complex double-precision
#elif PRECISION == 6464
  #define real dvec2
  struct real2 {real x; real y;};
  struct real4 {real x; real y; real z; real w;};
  struct real8 {real s0; real s1; real s2; real s3;
                  real s4; real s5; real s6; real s7;};
  struct real16 {real s0; real s1; real s2; real s3;
                 real s4; real s5; real s6; real s7;
                 real s8; real s9; real sA; real sB;
                 real sC; real sD; real sE; real sF;};
  #define ZERO 0.0
  #define ONE 1.0
  #define SMALLEST -1.0e37
#endif

// Single-element version of a complex number
#if PRECISION == 3232
  #define singlereal float
#elif PRECISION == 6464
  #define singlereal double
#else
  #define singlereal real
#endif

// Converts a 'real argument' value to a 'real' value as passed to the kernel. Normally there is no
// conversion, but half-precision is not supported as kernel argument so it is converted from float.
#if PRECISION == 16
  // this should hypothetically not be needed, since we can use Vulkan to bitcast whatever we want
  #define real_arg float
  #define GetRealArg(x) float16_t(x)
#else
  #define real_arg real
  #define GetRealArg(x) x
#endif

// Not sure how this will go together with shared memory. It remains to be seen...
// 		For now, I will use 'shared' as the equivalent, despite it not actually being a function argument
// Pointers to local memory objects (using a define because CUDA doesn't need them)
#ifndef LOCAL_PTR
  #define LOCAL_PTR shared
#endif

// =================================================================================================


// By default the workgroup size requirement is enabled. For Qualcomm devices the workgroup size 
// requirement results in worse performance and is disabled (src/utilities/compile.cpp)
//		in vulkan, this is simply not possible without a bunch of stupidity.
#ifndef RELAX_WORKGROUP_SIZE
  #define RELAX_WORKGROUP_SIZE 0
#endif

// Sets a variable to zero
#if PRECISION == 3232 || PRECISION == 6464
  #define SetToZero(a) a.x = ZERO; a.y = ZERO
#else
  #define SetToZero(a) a = ZERO
#endif

// Sets a variable to zero (only the imaginary part)
#if PRECISION == 3232 || PRECISION == 6464
  #define ImagToZero(a) a.y = ZERO
#else
  #define ImagToZero(a) 
#endif

// Sets a variable to one
#if PRECISION == 3232 || PRECISION == 6464
  #define SetToOne(a) a.x = ONE; a.y = ZERO
#else
  #define SetToOne(a) a = ONE
#endif

// Determines whether a variable is zero
#if PRECISION == 3232 || PRECISION == 6464
  #define IsZero(a) ((a.x == ZERO) && (a.y == ZERO))
#else
  #define IsZero(a) (a == ZERO)
#endif

// The absolute value (component-wise)
#if PRECISION == 3232 || PRECISION == 6464
  #define AbsoluteValue(value) value.x = abs(value.x); value.y = abs(value.y)
#else
  #define AbsoluteValue(value) value = abs(value)
#endif

// Negation (component-wise)
#if PRECISION == 3232 || PRECISION == 6464
  #define Negate(value) value.x = -(value.x); value.y = -(value.y)
#else
  #define Negate(value) value = -(value)
#endif

// Adds two complex variables
#if PRECISION == 3232 || PRECISION == 6464
  #define Add(c,a,b) c.x = a.x + b.x; c.y = a.y + b.y
#else
  #define Add(c,a,b) c = a + b
#endif

// Subtracts two complex variables
#if PRECISION == 3232 || PRECISION == 6464
  #define Subtract(c,a,b) c.x = a.x - b.x; c.y = a.y - b.y
#else
  #define Subtract(c,a,b) c = a - b
#endif

// Multiply two complex variables (used in the defines below)
#if PRECISION == 3232 || PRECISION == 6464
  #define MulReal(a,b) a.x*b.x - a.y*b.y
  #define MulImag(a,b) a.x*b.y + a.y*b.x
#endif

// The scalar multiply function
#if PRECISION == 3232 || PRECISION == 6464
  #define Multiply(c,a,b) c.x = MulReal(a,b); c.y = MulImag(a,b)
#else
  #define Multiply(c,a,b) c = a * b
#endif

// The scalar multiply-add function
#if PRECISION == 3232 || PRECISION == 6464
  #define MultiplyAdd(c,a,b) c.x += MulReal(a,b); c.y += MulImag(a,b)
#else
  #define MultiplyAdd(c,a,b) c += a * b
#endif

// The scalar multiply-subtract function
#if PRECISION == 3232 || PRECISION == 6464
  #define MultiplySubtract(c,a,b) c.x -= MulReal(a,b); c.y -= MulImag(a,b)
#else
  #define MultiplySubtract(c,a,b) c -= a * b
#endif

// The scalar division function: full division
#if PRECISION == 3232 || PRECISION == 6464
  #define DivideFull(c,a,b) singlereal num_x = (a.x * b.x) + (a.y * b.y); singlereal num_y = (a.y * b.x) - (a.x * b.y); singlereal denom = (b.x * b.x) + (b.y * b.y); c.x = num_x / denom; c.y = num_y / denom
#else
  #define DivideFull(c,a,b) c = a / b
#endif

// The scalar AXPBY function
#if PRECISION == 3232 || PRECISION == 6464
  #define AXPBY(e,a,b,c,d) e.x = MulReal(a,b) + MulReal(c,d); e.y = MulImag(a,b) + MulImag(c,d)
#else
  #define AXPBY(e,a,b,c,d) e = a*b + c*d
#endif

// The complex conjugate operation for complex transforms
#if PRECISION == 3232 || PRECISION == 6464
  #define COMPLEX_CONJUGATE(value) value.x = value.x; value.y = -value.y
#else
  #define COMPLEX_CONJUGATE(value) 
#endif

// =================================================================================================

// Shuffled workgroup indices to avoid partition camping, see below. For specific devices, this is
// enabled (see src/routine.cc).
#ifndef USE_STAGGERED_INDICES
  #define USE_STAGGERED_INDICES 0
#endif

// Staggered/shuffled group indices to avoid partition camping (AMD GPUs). Formula's are taken from:
// http://docs.nvidia.com/cuda/samples/6_Advanced/transpose/doc/MatrixTranspose.pdf
// More details: https://github.com/CNugteren/CLBlast/issues/53
#if USE_STAGGERED_INDICES == 1 && GEMMK == 0
  int GetGroupIDFlat() {
    return get_group_id(0) + get_num_groups(0) * get_group_id(1);
  }
  int GetGroupID1() {
    return (GetGroupIDFlat()) % get_num_groups(1);
  }
  int GetGroupID0() {
    return ((GetGroupIDFlat() / get_num_groups(1)) + GetGroupID1()) % get_num_groups(0);
  }
#else
  int GetGroupID1() { return int(gl_WorkGroupID.y); }
  int GetGroupID0() { return int(gl_WorkGroupID.x); }
#endif
