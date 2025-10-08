#include <iostream>
#include "tests-common.hpp"
#include "test-blas-runtime.hpp"

int main(int argc, char** argv)
{
	START_TEST("dummy");
	size_t a = 1;
	size_t b = 2;
	ASSERT_EQUAL(a, b);
}
