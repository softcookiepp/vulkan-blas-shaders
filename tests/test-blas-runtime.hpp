#ifndef TART_BLAS_FWD
#define TART_BLAS_FWD
#include "../tart/include/tart.hpp"
#include <cblas.h>
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

} // namespace tartblas

#endif
