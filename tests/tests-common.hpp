#ifndef TARTBLAS_TESTS_DOCTEST_INCLUDE
#define TARTBLAS_TESTS_DOCTEST_INCLUDE
#include <chrono>
#include <stdexcept>
#include <string>
#include <random>
#include <cstdlib>
#include <sstream>

std::random_device rd;
std::mt19937 gGenerator(rd());
std::normal_distribution<> gDist(-1.0, 1.0);

float randn()
{
	return gDist(gGenerator);
}

std::vector<float> randn(uint32_t size)
{
	std::vector<float> v(size);
	for (size_t i = 0; i < v.size(); i += 1)
	{
		v[i] = randn();
	}
	return v;
}

template <typename T>
void printVector(std::vector<T>& v)
{
	for (T& elem : v) std::cout << elem << ", ";
	std::cout << std::endl;
}

template <typename T>
void printMatrix(std::vector<T>&, uint32_t height, uint32_t width)
{
	
}

template <typename T>
void ASSERT_EQUAL(T& a, T& b)
{
	if (a != b) throw std::runtime_error("assertion failed!");
}

void ASSERT_CLOSE(float a, float b, float tolerance = 1.0e-5)
{
	if (abs(a - b) > tolerance)
	{
		std::stringstream ss;
		ss << "assertion failed: " << a << " is not close to "
			<< b << " (tolerance: " << tolerance << ")" << std::endl;
		throw std::runtime_error(ss.str());
	}
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
