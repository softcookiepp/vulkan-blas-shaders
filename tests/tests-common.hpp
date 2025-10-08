#ifndef TARTBLAS_TESTS_DOCTEST_INCLUDE
#define TARTBLAS_TESTS_DOCTEST_INCLUDE
#include <chrono>
#include <stdexcept>
#include <string>

template <typename T>
void ASSERT_EQUAL(T& a, T& b)
{
	if (a != b) throw std::runtime_error("assertion failed!");
}

void START_TEST(std::string title)
{
	std::cout << "\nStarting test: " << title << std::endl;
}

#endif
