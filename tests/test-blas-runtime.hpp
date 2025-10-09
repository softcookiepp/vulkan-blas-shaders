#ifndef TART_BLAS_FWD
#define TART_BLAS_FWD
#include "../tart/include/tart.hpp"
#include <openblas/cblas.h>
#include <map>
#include <string>

tart::Instance gTartInstance;

tart::device_ptr getTestDevice()
{
	return gTartInstance.createDevice(0);
}


namespace tartblas
{

// shader modules
std::map<std::string, tart::shader_module_ptr> gShaderModules;
tart::shader_module_ptr getShaderModule(std::string shaderPath)
{
	if (gShaderModules.find(shaderPath) == gShaderModules.end() )
		gShaderModules[shaderPath] = getTestDevice()->loadShaderFromPath(shaderPath);
	return gShaderModules[shaderPath];
}

std::map<tart::shader_module_ptr, std::map<std::vector<uint8_t>, tart::pipeline_ptr> > gShaderPipelines;
tart::pipeline_ptr getShaderPipeline(std::string shaderPath, std::vector<uint8_t> specConsts = {}, std::vector<uint8_t> pushConsts = {})
{
	tart::shader_module_ptr shaderModule = getShaderModule(shaderPath);
	if (gShaderPipelines.find(shaderModule) == gShaderPipelines.end() )
	{
		std::map<std::vector<uint8_t>, tart::pipeline_ptr> map;
		gShaderPipelines[shaderModule] = map;
	}
	std::map<std::vector<uint8_t>, tart::pipeline_ptr>& mapRef = gShaderPipelines[shaderModule];
	if (mapRef.find(specConsts) == mapRef.end())
		mapRef[specConsts] = getTestDevice()->createPipeline(shaderModule, "main", specConsts, pushConsts);
	return mapRef[specConsts];
}


// LEVEL 1 FUNCTIONS
void ssum(tart::command_sequence_ptr sequence, uint32_t size,
	tart::buffer_ptr x, int32_t incx, tart::buffer_ptr y)
{
	struct {
		uint32_t size;
		int32_t incx;
	} pushConstStruct = {size, incx};
	
	std::vector<uint8_t> packedPushConsts = tart::packConstants(pushConstStruct);
	tart::pipeline_ptr pipeline = getShaderPipeline("spv/sum.spv", {}, packedPushConsts);
	sequence->recordPipeline(pipeline, {1, 1, 1}, {x, y}, packedPushConsts);
}

void sdot(tart::command_sequence_ptr sequence, uint32_t size,
	tart::buffer_ptr x, int32_t incx,
	tart::buffer_ptr y, int32_t incy,
	tart::buffer_ptr out)
{
	struct {
		uint32_t size;
		int32_t incx;
		int32_t incy;
	} pushStruct = {size, incx, incy};
	if (pushStruct.size != size) throw std::runtime_error("this should not happen");
	std::vector<uint8_t> packedPushConsts = tart::packConstants(pushStruct);
	tart::pipeline_ptr pipeline = getShaderPipeline("spv/dot.spv", {}, packedPushConsts);
	sequence->recordPipeline(pipeline, {1, 1, 1}, {x, y, out}, packedPushConsts);
}

void sasum(tart::command_sequence_ptr sequence, uint32_t size,
	tart::buffer_ptr x, int32_t incx, tart::buffer_ptr y, bool sync = false)
{
	struct {
		uint32_t size;
		int32_t incx;
	} pushConstStruct = {size, incx};
	
	std::vector<uint8_t> packedPushConsts = tart::packConstants(pushConstStruct);
	tart::pipeline_ptr pipeline = getShaderPipeline("spv/asum.spv", {}, packedPushConsts);
	sequence->recordPipeline(pipeline, {1, 1, 1}, {x, y}, packedPushConsts);
}

void snrm2(tart::command_sequence_ptr sequence, uint32_t size,
	tart::buffer_ptr x, int32_t incx, tart::buffer_ptr y, bool sync = false)
{
	struct {
		uint32_t size;
		int32_t incx;
	} pushConstStruct = {size, incx};
	
	std::vector<uint8_t> packedPushConsts = tart::packConstants(pushConstStruct);
	tart::pipeline_ptr pipeline = getShaderPipeline("spv/nrm2.spv", {}, packedPushConsts);
	sequence->recordPipeline(pipeline, {1, 1, 1}, {x, y}, packedPushConsts);
}

void samax(tart::command_sequence_ptr sequence, uint32_t size,
	tart::buffer_ptr x, int32_t incx, tart::buffer_ptr y, bool sync = false)
{
	struct {
		uint32_t size;
		int32_t incx;
	} pushConstStruct = {size, incx};
	
	std::vector<uint8_t> packedPushConsts = tart::packConstants(pushConstStruct);
	tart::pipeline_ptr pipeline = getShaderPipeline("spv/amax.spv", {}, packedPushConsts);
	sequence->recordPipeline(pipeline, {1, 1, 1}, {x, y}, packedPushConsts);
}

void samin(tart::command_sequence_ptr sequence, uint32_t size,
	tart::buffer_ptr x, int32_t incx, tart::buffer_ptr y, bool sync = false)
{
	struct {
		uint32_t size;
		int32_t incx;
	} pushConstStruct = {size, incx};
	
	std::vector<uint8_t> packedPushConsts = tart::packConstants(pushConstStruct);
	tart::pipeline_ptr pipeline = getShaderPipeline("spv/amin.spv", {}, packedPushConsts);
	sequence->recordPipeline(pipeline, {1, 1, 1}, {x, y}, packedPushConsts);
}

void smax(tart::command_sequence_ptr sequence, uint32_t size,
	tart::buffer_ptr x, int32_t incx, tart::buffer_ptr y, bool sync = false)
{
	struct {
		uint32_t size;
		int32_t incx;
	} pushConstStruct = {size, incx};
	
	std::vector<uint8_t> packedPushConsts = tart::packConstants(pushConstStruct);
	tart::pipeline_ptr pipeline = getShaderPipeline("spv/max.spv", {}, packedPushConsts);
	sequence->recordPipeline(pipeline, {1, 1, 1}, {x, y}, packedPushConsts);
}

void smin(tart::command_sequence_ptr sequence, uint32_t size,
	tart::buffer_ptr x, int32_t incx, tart::buffer_ptr y, bool sync = false)
{
	struct {
		uint32_t size;
		int32_t incx;
	} pushConstStruct = {size, incx};
	
	std::vector<uint8_t> packedPushConsts = tart::packConstants(pushConstStruct);
	tart::pipeline_ptr pipeline = getShaderPipeline("spv/min.spv", {}, packedPushConsts);
	sequence->recordPipeline(pipeline, {1, 1, 1}, {x, y}, packedPushConsts);
}

void saxpy(tart::command_sequence_ptr sequence, uint32_t n,
	float alpha, 
	tart::buffer_ptr x, int32_t incx, tart::buffer_ptr y, int32_t incy)
{
	struct {
		uint32_t n;
		float alpha;
		int32_t incx;
		int32_t incy;
	} pushConstStruct = {n, alpha, incx, incy};
	
	std::vector<uint8_t> packedPushConsts = tart::packConstants(pushConstStruct);
	tart::pipeline_ptr pipeline = getShaderPipeline("spv/axpy.spv", {}, packedPushConsts);
	sequence->recordPipeline(pipeline, {n, 1, 1}, {x, y}, packedPushConsts);
}

void scopy(tart::command_sequence_ptr sequence, uint32_t n,
	tart::buffer_ptr x, int32_t incx,
	tart::buffer_ptr y, int32_t incy)
{
	struct {
		uint32_t n;
		int32_t incx;
		int32_t incy;
	} pushConstStruct = {n, incx, incy};
	
	std::vector<uint8_t> packedPushConsts = tart::packConstants(pushConstStruct);
	tart::pipeline_ptr pipeline = getShaderPipeline("spv/copy.spv", {}, packedPushConsts);
	sequence->recordPipeline(pipeline, {n, 1, 1}, {x, y}, packedPushConsts);
}

void sswap(tart::command_sequence_ptr sequence, uint32_t n,
	tart::buffer_ptr x, int32_t incx,
	tart::buffer_ptr y, int32_t incy)
{
	struct {
		uint32_t n;
		int32_t incx;
		int32_t incy;
	} pushConstStruct = {n, incx, incy};
	
	std::vector<uint8_t> packedPushConsts = tart::packConstants(pushConstStruct);
	tart::pipeline_ptr pipeline = getShaderPipeline("spv/swap.spv", {}, packedPushConsts);
	sequence->recordPipeline(pipeline, {n, 1, 1}, {x, y}, packedPushConsts);
}

void srot(tart::command_sequence_ptr sequence, uint32_t n,
	tart::buffer_ptr x, int32_t incx,
	tart::buffer_ptr y, int32_t incy,
	float c, float s)
{
	struct {
		uint32_t n;
		int32_t incx;
		int32_t incy;
		float c;
		float s;
	} pushConstStruct = {n, incx, incy, c, s};
	
	std::vector<uint8_t> packedPushConsts = tart::packConstants(pushConstStruct);
	tart::pipeline_ptr pipeline = getShaderPipeline("spv/rot.spv", {}, packedPushConsts);
	sequence->recordPipeline(pipeline, {n, 1, 1}, {x, y}, packedPushConsts);
}

void srotg(tart::command_sequence_ptr sequence,
	tart::buffer_ptr a,
	tart::buffer_ptr b,
	tart::buffer_ptr c,
	tart::buffer_ptr s)
{
	tart::pipeline_ptr pipeline = getShaderPipeline("spv/rotg.spv", {}, {});
	sequence->recordPipeline(pipeline, {1, 1, 1}, {a, b, c, s});
}

void srotm(tart::command_sequence_ptr sequence, uint32_t n,
	tart::buffer_ptr x, int32_t incx,
	tart::buffer_ptr y, int32_t incy,
	std::vector<float> param)
{
	if (param.size() != 5)
		throw std::invalid_argument("param size must be 5!");
	struct {
		uint32_t n;
		int32_t incx;
		int32_t incy;
		float p0;
		float p1;
		float p2;
		float p3;
		float p4;
	} pushConstStruct = {n, incx, incy, param[0], param[1], param[2], param[3], param[4]};
	
	std::vector<uint8_t> packedPushConsts = tart::packConstants(pushConstStruct);
	tart::pipeline_ptr pipeline = getShaderPipeline("spv/rotm.spv", {}, packedPushConsts);
	sequence->recordPipeline(pipeline, {n, 1, 1}, {x, y}, packedPushConsts);
}

void sscal(tart::command_sequence_ptr sequence, uint32_t n,
	float alpha, tart::buffer_ptr x, int32_t incx)
{
	struct {
		uint32_t n;
		float alpha;
		int32_t incx;
	} pushConstStruct = {n, alpha, incx};
	
	std::vector<uint8_t> packedPushConsts = tart::packConstants(pushConstStruct);
	tart::pipeline_ptr pipeline = getShaderPipeline("spv/scal.spv", {}, packedPushConsts);
	sequence->recordPipeline(pipeline, {n}, {x}, packedPushConsts);
}

// LEVEL 2
void sgemv(tart::command_sequence_ptr sequence,
	enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE transpose,
	uint32_t m, uint32_t n, float alpha,
	tart::buffer_ptr A, uint32_t lda,
	tart::buffer_ptr x, int32_t incx,
	float beta,
	tart::buffer_ptr y, int32_t incy)
{
	// assume a ground truth that all supplied arrays are inherently row major.
	// if supplied order is column major and no transpose is specified, then the array is transposed.
	// if row major and transpose is specified, then array is transposed.
	// in other cases, the array is not transposed.
	bool useTranspose = (order == CblasColMajor && transpose == CblasNoTrans)
		|| (order == CblasRowMajor && transpose == CblasTrans);
	//if (useTranspose)
	//	throw std::invalid_argument("using transpose is not yet validated!");
	
	if (transpose == CblasTrans)
	{
		uint32_t tmp = m;
		m = n;
		n = tmp;
	}
	
	struct {
		uint32_t use_transpose; // booleans in GLSL are 32-bit
		float alpha;
		uint32_t lda;
		int32_t incx;
		float beta;
		int32_t incy;
		uint32_t m;
		uint32_t n;
	} pushConstStruct = {(uint32_t)useTranspose, alpha, lda, incx, beta, incy, m, n};
	
	// each matrix row element will need to have a partial sum associated with it.
	// the row size is n
	// which means the local size should also be n.
	std::vector<uint8_t> packedPushConsts = tart::packConstants(pushConstStruct);
	struct {
		uint32_t n;
	} specConstStruct = {n};
	std::vector<uint8_t> packedSpecConsts = tart::packConstants(specConstStruct);
	tart::pipeline_ptr pipeline = getShaderPipeline("spv/gemv.spv", packedSpecConsts, packedPushConsts);
	sequence->recordPipeline(pipeline, {m}, {x, A, y}, packedPushConsts);
}

void sger(tart::command_sequence_ptr sequence,
	enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE transpose,
	uint32_t m, uint32_t n, float alpha,
	tart::buffer_ptr x, int32_t incx,
	tart::buffer_ptr y, int32_t incy,
	tart::buffer_ptr A, uint32_t lda)
{
	// assume a ground truth that all supplied arrays are inherently row major.
	// if supplied order is column major and no transpose is specified, then the array is transposed.
	// if row major and transpose is specified, then array is transposed.
	// in other cases, the array is not transposed.
	bool useTranspose = (order == CblasColMajor && transpose == CblasNoTrans)
		|| (order == CblasRowMajor && transpose == CblasTrans);

	
	
	struct {
		uint32_t use_transpose; // booleans in GLSL are 32-bit
		uint32_t m;
		uint32_t n;
		float alpha;
		int32_t incx;
		int32_t incy;
		uint32_t lda;
	} pushConstStruct = {(uint32_t)useTranspose, m, n, alpha, incx, incy, lda};
	
	std::vector<uint8_t> packedPushConsts = tart::packConstants(pushConstStruct);
	tart::pipeline_ptr pipeline = getShaderPipeline("spv/ger.spv", {}, packedPushConsts);
	sequence->recordPipeline(pipeline, {n, m}, {x, y, A}, packedPushConsts);
}

} // namespace tartblas

#endif
