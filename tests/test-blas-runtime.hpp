#ifndef TART_BLAS_FWD
#define TART_BLAS_FWD
#include "../tart/include/tart.hpp"
#include <openblas/cblas.h>
#include <map>
#include <string>

const uint32_t TRSV_BLOCK_SIZE = 32;

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
#if 0
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
	bool requiresFlip = (order == CblasColMajor && transpose == CblasNoTrans)
		|| (order == CblasRowMajor && transpose == CblasTrans);
		
	struct {
		uint32_t requires_flip; // booleans in GLSL are 32-bit
		float alpha;
		uint32_t lda;
		int32_t incx;
		float beta;
		int32_t incy;
		uint32_t m;
		uint32_t n;
	} pushConstStruct = {(uint32_t)requiresFlip, alpha, lda, incx, beta, incy, m, n};
	
	if (transpose == CblasTrans)
	{
		// flip m and n
		pushConstStruct.m = n;
		pushConstStruct.n = m;
	}
	
	// each matrix row element will need to have a partial sum associated with it.
	// the row size is n
	// which means the local size should also be n.
	std::vector<uint8_t> packedPushConsts = tart::packConstants(pushConstStruct);
	struct {
		uint32_t n;
	} specConstStruct = {pushConstStruct.n};
	std::vector<uint8_t> packedSpecConsts = tart::packConstants(specConstStruct);
	tart::pipeline_ptr pipeline = getShaderPipeline("spv/gemv-new.spv", packedSpecConsts, packedPushConsts);
	sequence->recordPipeline(pipeline, {pushConstStruct.m}, {x, A, y}, packedPushConsts);
}
#else
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
	bool requiresFlip = (order == CblasColMajor && transpose == CblasNoTrans)
		|| (order == CblasRowMajor && transpose == CblasTrans);
		
	struct {
		uint32_t requires_flip; // booleans in GLSL are 32-bit
		float alpha;
		uint32_t lda;
		int32_t incx;
		float beta;
		int32_t incy;
		uint32_t m;
		uint32_t n;
	} pushConstStruct = {(uint32_t)requiresFlip, alpha, lda, incx, beta, incy, m, n};
	
	if (transpose == CblasTrans)
	{
		// flip m and n
		pushConstStruct.m = n;
		pushConstStruct.n = m;
	}
	
	// each matrix row element will need to have a partial sum associated with it.
	// the row size is n
	// which means the local size should also be n.
	std::vector<uint8_t> packedPushConsts = tart::packConstants(pushConstStruct);
	struct {
		uint32_t n;
	} specConstStruct = {pushConstStruct.n};
	std::vector<uint8_t> packedSpecConsts = tart::packConstants(specConstStruct);
	tart::pipeline_ptr pipeline = getShaderPipeline("spv/gemv.spv", packedSpecConsts, packedPushConsts);
	sequence->recordPipeline(pipeline, {pushConstStruct.m}, {x, A, y}, packedPushConsts);
}
#endif

void sger(tart::command_sequence_ptr sequence,
	enum CBLAS_ORDER order,
	uint32_t m, uint32_t n, float alpha,
	tart::buffer_ptr x, int32_t incx,
	tart::buffer_ptr y, int32_t incy,
	tart::buffer_ptr A, uint32_t lda)
{
	// assume a ground truth that all supplied arrays are inherently row major.
	// if supplied order is column major and no transpose is specified, then the array is transposed.
	// if row major and transpose is specified, then array is transposed.
	// in other cases, the array is not transposed.
	bool requiresFlip = order == CblasColMajor;
	struct {
		uint32_t requires_flip; // booleans in GLSL are 32-bit
		uint32_t m;
		uint32_t n;
		float alpha;
		int32_t incx;
		int32_t incy;
		uint32_t lda;
	} pushConstStruct = {(uint32_t)requiresFlip, m, n, alpha, incx, incy, lda};
	
	std::vector<uint8_t> packedPushConsts = tart::packConstants(pushConstStruct);
	tart::pipeline_ptr pipeline = getShaderPipeline("spv/ger.spv", {}, packedPushConsts);
	sequence->recordPipeline(pipeline, {pushConstStruct.n, pushConstStruct.m}, {x, y, A}, packedPushConsts);
}

void fillVector(tart::command_sequence_ptr sequence, uint32_t n, uint32_t incx, tart::buffer_ptr xBuf, float value)
{
	struct {
		int n;
		int inc;
		int offset;
		float value;
	} pushStruct = {(int)n, (int)incx, 0, value};
	auto pushConstants = tart::packConstants(pushStruct);
	tart::pipeline_ptr pipeline = getShaderPipeline("spv/fill-vector.spv", {}, pushConstants);
	uint32_t global = (n / 16) + 1;
	sequence->recordPipeline(pipeline, {global}, {xBuf}, pushConstants);
}



#if 1
void strsv(tart::command_sequence_ptr sequence,
	const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans,
	const enum CBLAS_DIAG diag, uint32_t n, tart::buffer_ptr A, uint32_t lda, tart::buffer_ptr x, int32_t incx,
	tart::buffer_ptr xTmp)
{
	if (xTmp->getSize() < x->getSize()) throw std::invalid_argument("xTmp must be at least the size of x");
	
	// first copy x to xTmp
	sequence->recordCopyBuffer(xTmp, x);
	
	// fill xTmp with zeros
	fillVector(sequence, n, incx, xTmp, 0.0);
	
	bool isUpper = ((uplo == CblasUpper && trans == CblasNoTrans) ||
		(uplo == CblasLower && trans != CblasNoTrans));
	bool isTransposed = ((order == CblasColMajor && trans == CblasNoTrans) ||
		(order != CblasColMajor && trans != CblasNoTrans));
	bool isUnitDiagonal = (diag == CblasUnit);
	bool doConjugate = (trans == CblasConjTrans || CblasConjNoTrans == trans); // this should work?
	
	uint32_t col = n;
	for (size_t i = 0; i < n; i += TRSV_BLOCK_SIZE)
	{
		const uint32_t nmi = n - i;
		const uint32_t blockSize = std::min(TRSV_BLOCK_SIZE, nmi);
		col = (isUpper) ? col - blockSize : i;
		const auto extraOffsetA = (isTransposed) ? (isUpper ? col + (col + blockSize) * lda : col)
			: (isUpper ? col + blockSize + col * lda : col * lda);
		const auto extraOffsetXtmp = isUpper ? (col + blockSize)*incx : 0;
		const auto extraOffsetX = col*incx;
		const auto gemvM = (trans == CblasNoTrans) ? blockSize : i;
		const auto gemvN = (trans == CblasNoTrans) ? i : blockSize;
		if (i > 0)
		{
			// if this doesn't work, there may be unforseen problems with my own implementation...
			sgemv(sequence, order, trans, gemvM, gemvN, 1.0, A->view(extraOffsetA*sizeof(float) ), lda, xTmp->view(extraOffsetXtmp*sizeof(float)),
				incx, 1.0, xTmp->view(extraOffsetX*sizeof(float)), incx);
		}
		
		// now the actual substitution thingy
		
		// construct pipeline and push constants ahead of time
		struct {
			int n;
			// A
			int a_offset;
			int a_ld;
			// b
			int b_offset;
			int b_inc;
			// x
			int x_offset;
			int x_inc;
			int is_transposed;
			int is_unit_diagonal;
			int do_conjugate;
		} pushStruct = {(int)n, 0, (int)lda, 0, incx, 0, incx, (int)isTransposed, (int)isUnitDiagonal, (int)doConjugate};
		auto pushConsts = tart::packConstants(pushStruct);
		tart::pipeline_ptr pipeline = nullptr;
		if (isUpper)
			// backward
			// maybe I should change that thing into a spec constant?
			pipeline = getShaderPipeline("spv/trsv-backward.spv", {}, pushConsts);
		else
			// forward
			pipeline = getShaderPipeline("spv/trsv-forward.spv", {}, pushConsts);
		
		std::vector<uint32_t> wg = {(n / TRSV_BLOCK_SIZE) + 1};
		sequence->recordPipeline(pipeline, wg,
			{
				A->view( (col + col * lda)*sizeof(float) ),
				x->view(col*incx*sizeof(float)),
				xTmp->view(col*incx*sizeof(float) )
			}, pushConsts);
	}
	// oops, I forgot to copy
	sequence->recordCopyBuffer(x, xTmp);
}
#else
void strsv(tart::command_sequence_ptr sequence,
	const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo, const enum CBLAS_TRANSPOSE trans,
	const enum CBLAS_DIAG diag, uint32_t N, tart::buffer_ptr A, uint32_t lda, tart::buffer_ptr x, int32_t incx)
{
	
	if (diag == CblasUnit) throw std::runtime_error("unit diagonal triangulars are not implemented!");
	
	bool requiresFlip = (order == CblasColMajor && trans == CblasNoTrans)
		|| (order == CblasRowMajor && trans == CblasTrans);
	
	struct {
		uint32_t requires_flip; // booleans in GLSL are 32-bit
		uint32_t transpose; // also bool
		int32_t incx;
		uint32_t lda;
	} pushConstStruct = {(uint32_t)requiresFlip, (uint32_t)(trans == CblasTrans), incx, lda};
	std::vector<uint8_t> packedPushConsts = tart::packConstants(pushConstStruct);
	
	struct {
		uint32_t N;
		uint32_t LOWER;
		uint32_t UNIT_DIAGONAL;
	} specConstStruct = {N, (uint32_t)(uplo == CblasLower), (uint32_t)false};
	std::vector<uint8_t> packedSpecConsts = tart::packConstants(specConstStruct);
	tart::pipeline_ptr pipeline = getShaderPipeline("spv/trsv.spv", packedSpecConsts, packedPushConsts);
	sequence->recordPipeline(pipeline, {1, 1, 1}, {x ,A}, packedPushConsts);
}
#endif

} // namespace tartblas

#endif
