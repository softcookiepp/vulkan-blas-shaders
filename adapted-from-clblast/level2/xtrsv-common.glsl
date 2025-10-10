// =================================================================================================

// Parameters set by the tuner or by the database. Here they are given a basic default value in case
// this kernel file is used outside of the CLBlast library.


#ifndef TRSV_BLOCK_SIZE
  #define TRSV_BLOCK_SIZE 32    // The block size for forward or backward substition
#endif

// specify local size
layout(local_size_x = TRSV_BLOCK_SIZE, local_size_y = 1, local_size_z = 1) in;
