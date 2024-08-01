#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

cmake_minimum_required(VERSION 3.20)

#project(DiffRast LANGUAGES CUDA CXX)
project(DiffRast LANGUAGES HIP CXX)
#project(DiffRast)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
# Find hip
find_package(hip)
find_package (gflags REQUIRED)
# message(STATUS "GFLAGS_INCLUDE_DIR: ${GFLAGS_INCLUDE_DIR}")
# message(STATUS "GFLAGS_LIBRARIES: ${GFLAGS_LIBRARIES}")
# include_directories (${GFLAGS_INCLUDE_DIR})
find_package(glog REQUIRED)
# message(STATUS "GLOG_INCLUDE_DIR: ${GLOG_INCLUDE_DIR}")
# message(STATUS "GLOG_LIBRARIES: ${GLOG_LIBRARIES}")
# include_directories (${GLOG_INCLUDE_DIR})

# Set compiler and linker
set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})

set(CMAKE_CXX_LINKER   ${HIP_HIPCC_EXECUTABLE})
add_compile_options("-mprintf-kind=buffered")

add_library(CudaRasterizer
	cuda_rasterizer/backward.h
	cuda_rasterizer/backward.cu
	cuda_rasterizer/forward.h
	cuda_rasterizer/forward.cu
	cuda_rasterizer/auxiliary.h
	cuda_rasterizer/rasterizer_impl.cu
	cuda_rasterizer/rasterizer_impl.h
	cuda_rasterizer/rasterizer.h
)
#add_executable (foo main.cc)

#set_target_properties(CudaRasterizer PROPERTIES CUDA_ARCHITECTURES "75;86")
set_target_properties(CudaRasterizer PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(CudaRasterizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cuda_rasterizer)
target_include_directories(CudaRasterizer PRIVATE third_party/glm ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

#target_link_libraries (CudaRasterizer gflags glog)
#target_link_libraries (CudaRasterizer ${GFLAGS_LIBRARIES} ${GLOG_LIBRARIES})
#target_link_libraries (foo ${GLOG_LIBRARIES})