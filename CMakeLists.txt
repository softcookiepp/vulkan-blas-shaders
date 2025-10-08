cmake_minimum_required(VERSION 3.15)

project(vulkan-blas-test)
add_subdirectory(tart)

find_package(BLAS REQUIRED)

include_directories(include)
include_directories(SPIRV-Reflect)

macro(compile_glsl)
	add_custom_command(
		OUTPUT ${ARGV2}
		COMMAND glslc -fshader-stage=compute ${ARGV1} -o ${ARGV2}
		DEPENDS ${ARGV1}
		WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
		COMMENT "Buiding GLSL Shader"
	)
	add_custom_target(${ARGV0} DEPENDS ${ARGV2})
endmacro()

add_executable(test-blas test-blas.cpp)
target_link_libraries(test-blas PUBLIC tart)
target_link_libraries(test-blas PRIVATE BLAS::BLAS)
