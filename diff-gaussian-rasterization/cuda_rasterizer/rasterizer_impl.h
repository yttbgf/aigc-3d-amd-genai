/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#pragma once

#include <iostream>
#include <vector>
#include "rasterizer.h"
#include <cuda_runtime_api.h>

namespace CudaRasterizer
{
	template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	{
		if (chunk == nullptr) {
			//printf("obtain chunk == nullptr!\n");
		} else {
			//printf("obtain chunk:%p\n",  (void*)chunk);
		}
		//printf("obtain chunk:%p\n", (void*)chunk);
		//printf("obtain chunk:%zu\n", reinterpret_cast<std::uintptr_t>(chunk));
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1); //after alignment ptr's new address
		//printf("obtain offset:%zu\n", (void*)offset);
		ptr = reinterpret_cast<T*>(offset); //update old address to new address
		chunk = reinterpret_cast<char*>(ptr + count); //continous chunk update latest boundary to new data end
	}

	struct GeometryState
	{
		size_t scan_size;
		float* depths;
		char* scanning_space;
		bool* clamped;
		int* internal_radii;
		float2* means2D;
		float* cov3D;
		float4* conic_opacity;
		float* rgb;
		//u_int64_t* point_offsets;
		//u_int64_t* tiles_touched;
		uint32_t* point_offsets;
		uint32_t* tiles_touched;		

		static GeometryState fromChunk(char*& chunk, size_t P);
	};

	struct ImageState
	{
		uint2* ranges;
		uint32_t* n_contrib;

		static ImageState fromChunk(char*& chunk, size_t N);
	};

	struct BinningState
	{
		size_t sorting_size;
		uint64_t* point_list_keys_unsorted;
		uint64_t* point_list_keys;
		uint32_t* point_list_unsorted;
		uint32_t* point_list;
		char* list_sorting_space;

		static BinningState fromChunk(char*& chunk, size_t P);
	};

	template<typename T> 
	size_t required(size_t P)
	{//get T chunk_size
		char* size = nullptr;
		T::fromChunk(size, P);
		//printf("size:%s\n", size);
		//printf("size: %zu\n", (size_t)size);
		//printf("size: %s %zu\n", size, (size_t)size);
		return ((size_t)size) + 128;
	}
};
