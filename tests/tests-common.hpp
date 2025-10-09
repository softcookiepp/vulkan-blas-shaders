#ifndef TARTBLAS_TESTS_DOCTEST_INCLUDE
#define TARTBLAS_TESTS_DOCTEST_INCLUDE
#include <chrono>
#include <stdexcept>
#include <string>
#include <random>
#include <cstdlib>

std::random_device rd;
std::mt19937 gGenerator(rd());
std::normal_distribution<> gDist(-1.0, 1.0);

std::vector<float> randn(uint32_t size)
{
	std::vector<float> v(size);
	for (size_t i = 0; i < v.size(); i += 1)
	{
		v[i] = gDist(gGenerator);
	}
	return v;
}

template <typename T>
void ASSERT_EQUAL(T& a, T& b)
{
	if (a != b) throw std::runtime_error("assertion failed!");
}

void ASSERT_CLOSE(float a, float b, float tolerance = 1.0e-5)
{
	if (abs(a - b) > tolerance) throw std::runtime_error("assertion failed!");
}

void ASSERT_CLOSE(std::vector<float>& a, std::vector<float>& b, float tolerance = 1.0e-5)
{
	if (a.size() != b.size()) throw std::runtime_error("size mismatch!");
	for (size_t i = 0; i < a.size(); i += 1)
	{
		ASSERT_CLOSE(a[i], b[i]);
	}
}

void START_TEST(std::string title)
{
	std::cout << "\nStarting test: " << title << std::endl;
}

#endif
