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

float invokeSum(uint32_t size, tart::buffer_ptr x, int32_t incx, tart::buffer_ptr y = nullptr, bool sync = false)
{
	struct {
		uint32_t size = size;
		int32_t incx = incx;
	} pushConsts;
	
	
	std::vector<uint8_t> packedPushConsts = tart::packConstants(pushConsts);
	tart::pipeline_ptr sumPipeline = getShaderPipeline("spv/sum.spv", pushConsts = pushConsts);
	sumPipeline->dispatch({1, 1, 1}, {}
	return 0.0;
}

} // namespace tartblas

#endif
