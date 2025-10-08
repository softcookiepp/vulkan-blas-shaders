#ifndef TARTBLAS_TESTS_DOCTEST_INCLUDE
#define TARTBLAS_TESTS_DOCTEST_INCLUDE
#include <chrono>
//#include "../doctest/doctest/doctest.h"
#include <stdexcept>
#include "../tart/include/tart.hpp"

tart::Instance gTartInstance;

tart::device_ptr getTestDevice()
{
	return gTartInstance.createDevice(0);
}

#endif
