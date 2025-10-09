#include <iostream>
#include "tests-common.hpp"
#include "test-blas-runtime.hpp"

void testSum()
{
	START_TEST("sum");
	const uint32_t SIZE = 2345;
	tart::device_ptr dev = getTestDevice();
	
	std::vector<float> x = randn(SIZE);
	tart::buffer_ptr xBuf = dev->allocateBuffer(x);
	tart::buffer_ptr s = dev->allocateBuffer(sizeof(float));
	
	tart::command_sequence_ptr sequence = dev->createSequence();
	tartblas::ssum(sequence, SIZE, xBuf, 1, s);
	dev->submitSequence(sequence);
	dev->sync();
	
	float expected = cblas_ssum(SIZE, x.data(), 1);
	std::vector<float> vResult = s->copyOut<float>();
	float result = vResult[0];
	ASSERT_CLOSE(expected, result);
}

void testDot()
{
	START_TEST("dot");
	const uint32_t SIZE = 2345;
	tart::device_ptr dev = getTestDevice();
	
	std::vector<float> x = randn(SIZE);
	std::vector<float> y = randn(SIZE);
	tart::buffer_ptr xBuf = dev->allocateBuffer(x);
	tart::buffer_ptr yBuf = dev->allocateBuffer(y);
	tart::buffer_ptr s = dev->allocateBuffer(sizeof(float));
	
	tart::command_sequence_ptr sequence = dev->createSequence();
	tartblas::sdot(sequence, SIZE, xBuf, 1, yBuf, -1, s);
	dev->submitSequence(sequence);
	dev->sync();
	
	float expected = cblas_sdot(SIZE, x.data(), 1, y.data(), -1);
	std::vector<float> vResult = s->copyOut<float>();
	float result = vResult[0];
	ASSERT_CLOSE(expected, result);
}

void testAsum()
{
	START_TEST("asum");
	const uint32_t SIZE = 2345;
	tart::device_ptr dev = getTestDevice();
	
	std::vector<float> x = randn(SIZE);
	tart::buffer_ptr xBuf = dev->allocateBuffer(x);
	tart::buffer_ptr s = dev->allocateBuffer(sizeof(float));
	
	tart::command_sequence_ptr sequence = dev->createSequence();
	tartblas::sasum(sequence, SIZE, xBuf, 1, s);
	dev->submitSequence(sequence);
	dev->sync();
	
	float expected = cblas_sasum(SIZE, x.data(), 1);
	std::vector<float> vResult = s->copyOut<float>();
	float result = vResult[0];
	ASSERT_CLOSE(expected, result);
}

void testAmax()
{
	START_TEST("amax");
	const uint32_t SIZE = 2345;
	tart::device_ptr dev = getTestDevice();
	
	std::vector<float> x = randn(SIZE);
	tart::buffer_ptr xBuf = dev->allocateBuffer(x);
	tart::buffer_ptr s = dev->allocateBuffer(sizeof(float));
	
	tart::command_sequence_ptr sequence = dev->createSequence();
	tartblas::samax(sequence, SIZE, xBuf, 1, s);
	dev->submitSequence(sequence);
	dev->sync();
	
	uint32_t expected = (uint32_t)cblas_isamax(SIZE, x.data(), 1);
	std::vector<uint32_t> vResult = s->copyOut<uint32_t>();
	uint32_t result = vResult[0];
	ASSERT_EQUAL(expected, result);
}

void testAmin()
{
	START_TEST("amin");
	const uint32_t SIZE = 2345;
	tart::device_ptr dev = getTestDevice();
	
	std::vector<float> x = randn(SIZE);
	tart::buffer_ptr xBuf = dev->allocateBuffer(x);
	tart::buffer_ptr s = dev->allocateBuffer(sizeof(float));
	
	tart::command_sequence_ptr sequence = dev->createSequence();
	tartblas::samin(sequence, SIZE, xBuf, 1, s);
	dev->submitSequence(sequence);
	dev->sync();
	
	uint32_t expected = (uint32_t)cblas_isamin(SIZE, x.data(), 1);
	std::vector<uint32_t> vResult = s->copyOut<uint32_t>();
	uint32_t result = vResult[0];
	ASSERT_EQUAL(expected, result);
}

void testMax()
{
	START_TEST("max");
	const uint32_t SIZE = 2345;
	tart::device_ptr dev = getTestDevice();
	
	std::vector<float> x = randn(SIZE);
	tart::buffer_ptr xBuf = dev->allocateBuffer(x);
	tart::buffer_ptr s = dev->allocateBuffer(sizeof(float));
	
	tart::command_sequence_ptr sequence = dev->createSequence();
	tartblas::smax(sequence, SIZE, xBuf, 1, s);
	dev->submitSequence(sequence);
	dev->sync();
	
	uint32_t expected = (uint32_t)cblas_ismax(SIZE, x.data(), 1);
	std::vector<uint32_t> vResult = s->copyOut<uint32_t>();
	uint32_t result = vResult[0];
	ASSERT_EQUAL(expected, result);
}

void testMin()
{
	START_TEST("min");
	const uint32_t SIZE = 2345;
	tart::device_ptr dev = getTestDevice();
	
	std::vector<float> x = randn(SIZE);
	tart::buffer_ptr xBuf = dev->allocateBuffer(x);
	tart::buffer_ptr s = dev->allocateBuffer(sizeof(float));
	
	tart::command_sequence_ptr sequence = dev->createSequence();
	tartblas::smin(sequence, SIZE, xBuf, 1, s);
	dev->submitSequence(sequence);
	dev->sync();
	
	uint32_t expected = (uint32_t)cblas_ismin(SIZE, x.data(), 1);
	std::vector<uint32_t> vResult = s->copyOut<uint32_t>();
	uint32_t result = vResult[0];
	ASSERT_EQUAL(expected, result);
}

void testAxpy()
{
	START_TEST("axpy");
	const uint32_t SIZE = 2345;
	tart::device_ptr dev = getTestDevice();
	
	std::vector<float> x = randn(SIZE);
	std::vector<float> y = randn(SIZE);
	std::vector<float> alpha = randn(1);
	tart::buffer_ptr xBuf = dev->allocateBuffer(x);
	tart::buffer_ptr yBuf = dev->allocateBuffer(y);
	
	tart::command_sequence_ptr sequence = dev->createSequence();
	tartblas::saxpy(sequence, SIZE, alpha[0], xBuf, 1, yBuf, -1);
	dev->submitSequence(sequence);
	dev->sync();
	
	cblas_saxpy(SIZE, alpha[0], x.data(), 1, y.data(), -1);
	std::vector<float> vResult = yBuf->copyOut<float>();
	ASSERT_CLOSE(y, vResult);
}

void testCopy()
{
	START_TEST("copy");
	const uint32_t SIZE = 2345;
	tart::device_ptr dev = getTestDevice();
	
	std::vector<float> x = randn(SIZE);
	std::vector<float> y = randn(SIZE);
	tart::buffer_ptr xBuf = dev->allocateBuffer(x);
	tart::buffer_ptr yBuf = dev->allocateBuffer(y);
	
	tart::command_sequence_ptr sequence = dev->createSequence();
	tartblas::scopy(sequence, SIZE, xBuf, 1, yBuf, -1);
	dev->submitSequence(sequence);
	dev->sync();
	
	cblas_scopy(SIZE, x.data(), 1, y.data(), -1);
	std::vector<float> vResult = yBuf->copyOut<float>();
	ASSERT_CLOSE(y, vResult);
}

int main(int argc, char** argv)
{
	testSum();
	testDot();
	testAsum();
	testAmax();
	testMax();
	testAmin();
	testMin();
	testAxpy();
	testCopy();
}
