cmake_minimum_required(VERSION 3.15)

project(vulkan-blas-test)
add_subdirectory(tart)

find_package(BLAS REQUIRED)

include_directories(include)
include_directories(SPIRV-Reflect)

macro(compile_glsl)
	make_directory("${PROJECT_BINARY_DIR}/spv")
	execute_process(
		COMMAND glslc -fshader-stage=compute "${CMAKE_SOURCE_DIR}/${ARGV0}" -o "${PROJECT_BINARY_DIR}/spv/${ARGV1}"
		WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
	)
endmacro()

# compile all the shader binaries ahead of time
# level 1
compile_glsl(sum.glsl sum.spv)
compile_glsl(dot.glsl dot.spv)
compile_glsl(asum.glsl asum.spv)
compile_glsl(nrm2.glsl nrm2.spv)
compile_glsl(amax.glsl amax.spv)
compile_glsl(amin.glsl amin.spv)
compile_glsl(max.glsl max.spv)
compile_glsl(min.glsl min.spv)
compile_glsl(axpy.glsl axpy.spv)
compile_glsl(copy.glsl copy.spv)
compile_glsl(swap.glsl swap.spv)
compile_glsl(rot.glsl rot.spv)
compile_glsl(rotg.glsl rotg.spv)
compile_glsl(rotm.glsl rotm.spv)
# TODO: implement rotmg
compile_glsl(scal.glsl scal.spv)

# level 2
compile_glsl(gemv.glsl gemv.spv)
compile_glsl(ger.glsl ger.spv)
compile_glsl(trsv-lower.glsl trsv-lower.spv)
# thats all i got so far

add_executable(test-blas tests/test-blas.cpp)
target_link_libraries(test-blas PRIVATE tart)
target_link_libraries(test-blas PRIVATE BLAS::BLAS)
