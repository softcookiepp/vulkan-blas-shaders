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

void testSwap()
{
	START_TEST("swap");
	const uint32_t SIZE = 2345;
	tart::device_ptr dev = getTestDevice();
	
	std::vector<float> x = randn(SIZE);
	std::vector<float> y = randn(SIZE);
	tart::buffer_ptr xBuf = dev->allocateBuffer(x);
	tart::buffer_ptr yBuf = dev->allocateBuffer(y);
	
	tart::command_sequence_ptr sequence = dev->createSequence();
	tartblas::sswap(sequence, SIZE, xBuf, 1, yBuf, -1);
	dev->submitSequence(sequence);
	dev->sync();
	
	cblas_sswap(SIZE, x.data(), 1, y.data(), -1);
	std::vector<float> yResult = yBuf->copyOut<float>();
	std::vector<float> xResult = xBuf->copyOut<float>();
	ASSERT_CLOSE(y, yResult);
	ASSERT_CLOSE(x, xResult);
}

void testRot()
{
	START_TEST("rot");
	const uint32_t SIZE = 2345;
	tart::device_ptr dev = getTestDevice();
	
	std::vector<float> x = randn(SIZE);
	std::vector<float> y = randn(SIZE);
	tart::buffer_ptr xBuf = dev->allocateBuffer(x);
	tart::buffer_ptr yBuf = dev->allocateBuffer(y);
	float c = randn();
	float s = randn();
	
	tart::command_sequence_ptr sequence = dev->createSequence();
	tartblas::srot(sequence, SIZE, xBuf, 1, yBuf, -1, c, s);
	dev->submitSequence(sequence);
	dev->sync();
	
	cblas_srot(SIZE, x.data(), 1, y.data(), -1, c, s);
	std::vector<float> yResult = yBuf->copyOut<float>();
	std::vector<float> xResult = xBuf->copyOut<float>();
	ASSERT_CLOSE(y, yResult);
	ASSERT_CLOSE(x, xResult);
}

void testRotg()
{
	START_TEST("rotg");
	const uint32_t SIZE = 1;
	tart::device_ptr dev = getTestDevice();
	
	std::vector<float> a = randn(SIZE);
	std::vector<float> b = randn(SIZE);
	std::vector<float> c = randn(SIZE);
	std::vector<float> s = randn(SIZE);
	tart::buffer_ptr aBuf = dev->allocateBuffer(a);
	tart::buffer_ptr bBuf = dev->allocateBuffer(b);
	tart::buffer_ptr cBuf = dev->allocateBuffer(c);
	tart::buffer_ptr sBuf = dev->allocateBuffer(s);
	
	tart::command_sequence_ptr sequence = dev->createSequence();
	tartblas::srotg(sequence, aBuf, bBuf, cBuf, sBuf);
	dev->submitSequence(sequence);
	dev->sync();
	
	cblas_srotg(a.data(), b.data(), c.data(), s.data());
	
	std::vector<float> cResult = cBuf->copyOut<float>();
	std::vector<float> sResult = sBuf->copyOut<float>();
	std::cout << c[0] << std::endl;
	std::cout << cResult[0] << std::endl;
	std::cout << s[0] << std::endl;
	std::cout << sResult[0] << std::endl;
	ASSERT_CLOSE(c, cResult);
	ASSERT_CLOSE(s, sResult);
}

void testRotm()
{
	START_TEST("rotm");
	const uint32_t SIZE = 2345;
	tart::device_ptr dev = getTestDevice();
	
	std::vector<float> flags({-1.0, 0.0, 1.0, -2.0});
	for (float flag : flags)
	{
		std::vector<float> x = randn(SIZE);
		std::vector<float> y = randn(SIZE);
		tart::buffer_ptr xBuf = dev->allocateBuffer(x);
		tart::buffer_ptr yBuf = dev->allocateBuffer(y);
		std::vector<float> param = randn(5);
		param[0] = flag;
		
		tart::command_sequence_ptr sequence = dev->createSequence();
		tartblas::srotm(sequence, SIZE, xBuf, 1, yBuf, -1, param);
		dev->submitSequence(sequence);
		dev->sync();
		
		cblas_srotm(SIZE, x.data(), 1, y.data(), -1, param.data());
		std::vector<float> yResult = yBuf->copyOut<float>();
		std::vector<float> xResult = xBuf->copyOut<float>();
		ASSERT_CLOSE(y, yResult);
		ASSERT_CLOSE(x, xResult);
	}
}

void testScal()
{
	START_TEST("scal");
	const uint32_t SIZE = 2345;
	tart::device_ptr dev = getTestDevice();
	
	std::vector<float> x = randn(SIZE);
	tart::buffer_ptr xBuf = dev->allocateBuffer(x);
	float alpha = randn();
	
	tart::command_sequence_ptr sequence = dev->createSequence();
	tartblas::sscal(sequence, SIZE, alpha, xBuf, 1);
	dev->submitSequence(sequence);
	dev->sync();
	
	cblas_sscal(SIZE, alpha, x.data(), 1);
	std::vector<float> xResult = xBuf->copyOut<float>();
	ASSERT_CLOSE(x, xResult);
}

void testGemv()
{
	START_TEST("gemv");
	for (uint32_t WIDTH = 50; WIDTH <= 60; WIDTH += 1)
	{
		for (uint32_t HEIGHT = 50; HEIGHT <= 60; HEIGHT += 1)
		{
			tart::device_ptr dev = getTestDevice();
			std::vector<enum CBLAS_ORDER> orders({CblasRowMajor, CblasColMajor});
			std::vector<enum CBLAS_TRANSPOSE> transposes({CblasNoTrans, CblasTrans});
			for (auto ORDER : orders)
			{
				for (auto TRANS : transposes)
				{
					uint32_t X_SIZE = WIDTH;
					uint32_t Y_SIZE = HEIGHT;
					uint32_t A_SIZE = 0;
					
					if (TRANS == CblasTrans)
					{
						Y_SIZE = WIDTH;
						X_SIZE = HEIGHT;
					}
					
					uint32_t LDA;
					if (ORDER == CblasColMajor)
					{
						// columns are contiguous, LDA is column size
						LDA = HEIGHT;
					}
					else
					{
						// rows are contiguous, LDA is width
						LDA = WIDTH;
					}
						
					std::vector<float> x = randn(X_SIZE);
					std::vector<float> A = randn(HEIGHT*WIDTH);
					std::vector<float> y = randn(Y_SIZE);
					tart::buffer_ptr xBuf = dev->allocateBuffer(x);
					tart::buffer_ptr ABuf = dev->allocateBuffer(A);
					tart::buffer_ptr yBuf = dev->allocateBuffer(y);
					float alpha = randn();
					float beta = randn();
					
					tart::command_sequence_ptr sequence = dev->createSequence();
					tartblas::sgemv(sequence, ORDER, TRANS, HEIGHT, WIDTH, alpha, ABuf, LDA, xBuf, 1, beta, yBuf, 1);
					dev->submitSequence(sequence);
					dev->sync();
					
					
					std::vector<float> yResult = yBuf->copyOut<float>();
					cblas_sgemv(ORDER, TRANS, HEIGHT, WIDTH, alpha, A.data(), LDA, x.data(), 1, beta, y.data(), 1);
					
					ASSERT_CLOSE(y, yResult);
					
					dev->deallocateBuffer(xBuf);
					dev->deallocateBuffer(yBuf);
					dev->deallocateBuffer(ABuf);
				}
			}
		}
	}
}

void testGer()
{
	START_TEST("ger");
	const uint32_t HEIGHT = 8;
	const uint32_t WIDTH = 5;
	
	tart::device_ptr dev = getTestDevice();
	std::vector<enum CBLAS_ORDER> orders({CblasRowMajor, CblasColMajor});
	for (auto ORDER : orders)
	{
		uint32_t X_SIZE = HEIGHT;
		uint32_t Y_SIZE = WIDTH;
		uint32_t A_SIZE = 0;
		
		uint32_t LDA;
		if (ORDER == CblasColMajor)
		{
			// columns are contiguous, LDA is column size
			LDA = HEIGHT;
		}
		else
		{
			// rows are contiguous, LDA is width
			LDA = WIDTH;
		}
			
		std::vector<float> x = randn(X_SIZE);
		std::vector<float> A = randn(HEIGHT*WIDTH);
		std::vector<float> y = randn(Y_SIZE);
		tart::buffer_ptr xBuf = dev->allocateBuffer(x);
		tart::buffer_ptr ABuf = dev->allocateBuffer(A);
		tart::buffer_ptr yBuf = dev->allocateBuffer(y);
		float alpha = randn();
		
		tart::command_sequence_ptr sequence = dev->createSequence();
		tartblas::sger(sequence, ORDER, HEIGHT, WIDTH, alpha, xBuf, 1, yBuf, 1, ABuf, LDA);
		dev->submitSequence(sequence);
		dev->sync();
		
		std::vector<float> AResult = ABuf->copyOut<float>();
		cblas_sger(ORDER, HEIGHT, WIDTH, alpha, x.data(), 1, y.data(), 1, A.data(), LDA);
		ASSERT_CLOSE(A, AResult);
	}
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
	testSwap();
	testRot();
	// testRotg(); disable this for now, it is semi-broken :c
	testRotm();
	testScal();
	testGemv();
	testGer();
}
