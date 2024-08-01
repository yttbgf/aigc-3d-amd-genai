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

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
//#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float* depths,
	//const uint64_t* offsets,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	//std::cout << "duplicateWithKeys idx: " << idx << std::endl;
	//printf("duplicateWithKeys idx: %d \n", idx);
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t*)&depths[idx]);
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	//printf("identifyTileRanges idx:%d L:%d\n", L, idx);
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	//printf("identifyTileRanges currtile:%d \n", currtile);
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		//printf("identifyTileRanges  prevtile:%d\n", prevtile);
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum <<<(P + 255) / 256, 256 >>> (
	//hipLaunchKernelGGL(checkFrustum, dim3((P + 255) / 256), dim3(256), 0, 0,
	//checkFrustum <<< dim3((P + 255) / 256), dim3(256), 0, hipStreamDefault >>> (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	//printf("GeometryState obtain geom.depths\n");
	obtain(chunk, geom.depths, P, 128);
	//printf("GeometryState obtain geom.clamped\n");
	obtain(chunk, geom.clamped, P * 3, 128);
	//printf("GeometryState obtain geom.internal_radii\n");
	obtain(chunk, geom.internal_radii, P, 128);
	//printf("GeometryState obtain geom.means2D\n");
	obtain(chunk, geom.means2D, P, 128);
	//printf("GeometryState obtain geom.cov3D\n");
	obtain(chunk, geom.cov3D, P * 6, 128);
	//printf("GeometryState obtain geom.conic_opacity\n");
	obtain(chunk, geom.conic_opacity, P, 128);
	//printf("GeometryState obtain geom.depths\rgbn");
	obtain(chunk, geom.rgb, P * 3, 128);
	//printf("GeometryState obtain geom.tiles_touched\n");
	obtain(chunk, geom.tiles_touched, P, 128);
	//printf("GeometryState fromChunk geom.scan_size before:%zu\n", geom.scan_size);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P); //non sum just calc temp storage size
	//printf("fromChunk geom.scan_size after:%zu\n", geom.scan_size);
	//printf("GeometryState obtain geom.scanning_space\n");
	obtain(chunk, geom.scanning_space, geom.scan_size, 128); //malloc temp storage
	//printf("GeometryState obtain geom.point_offsets\n");
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	//printf("ImageState obtain img.n_contrib\n");
	obtain(chunk, img.n_contrib, N, 128);
	//printf("ImageState obtain img.ranges\n");
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	//printf("BinningState obtain binning.point_list\n");
	obtain(chunk, binning.point_list, P, 128);
	//printf("BinningState obtain binning.point_list_unsorted\n");	
	obtain(chunk, binning.point_list_unsorted, P, 128);
	//printf("BinningState obtain binning.point_list_keys\n");	
	obtain(chunk, binning.point_list_keys, P, 128);
	//printf("BinningState obtain binning.point_list_keys_unsorted\n");	
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	//printf("SortPairs...\n");
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	//printf("SortPairs finish!\n");
	//printf("BinningState obtain binning.list_sorting_space\n");	
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	//printf("BinningState return!\n");
	return binning;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P, int D, int M,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	float* out_color,
	float* out_depth,
	float* out_alpha,
	int* radii,
	bool debug)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	printf("required GeometryState\n");
	size_t chunk_size = required<GeometryState>(P);//only get GeometryState chunk_size
	printf("geometryBuffer chunk_size：%zu\n", chunk_size);
	char* chunkptr = geometryBuffer(chunk_size); // lambda template function tensor resize to real size  and return data ptr
	printf("fromChunk geomState\n");
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);//duplicated with required<GeometryState>?  chunkptr not nullptr but required is nullptr

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	printf("required imgState\n");
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	printf("fromChunk imgState width:%d height:%d w*h:%d img_chunk_size:%zu\n", width, height, width * height, img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	printf("preprocess before\n");
	/*
	printf("InclusiveSum geomState depths:%ld\n", geomState.depths);//%d out of range print negative
	printf("InclusiveSum geomState point_offsets:%ld\n", geomState.point_offsets);
	printf("InclusiveSum geomState tiles_touched:%ld\n", geomState.tiles_touched);
	printf("InclusiveSum geomState scanning_space:%ld\n", geomState.scanning_space);
	printf("InclusiveSum geomState clamped:%ld\n", geomState.clamped);
	printf("InclusiveSum geomState internal_radii:%ld\n", geomState.internal_radii);	
	printf("InclusiveSum geomState means2D:%ld\n", geomState.means2D);
	printf("InclusiveSum geomState cov3D:%ld\n", geomState.cov3D);	
	printf("InclusiveSum geomState conic_opacity:%ld\n", geomState.conic_opacity);
	printf("InclusiveSum geomState rgb:%ld\n", geomState.rgb);
	*/
	/*	
	// 在 CPU 上分配内存以存储 point_offsets 的副本
	int numPoints = P;
	uint32_t* point_offsets_host = new uint32_t[numPoints];

	// 从 GPU 复制数据到 CPU
	cudaMemcpy(point_offsets_host, geomState.point_offsets, numPoints * sizeof(uint32_t), cudaMemcpyDeviceToHost);

	// 打印 point_offsets_host 中的所有值
	for (size_t i = 0; i < numPoints; ++i) {
		printf("geomState.point_offsets[%zu]: %u\n", i, point_offsets_host[i]);
	}

	// 释放 CPU 上的内存
	delete[] point_offsets_host;
	*/

	int numPoints = P;//P / 10;
	/*
	//uint32_t* tiles_touched_host = new uint32_t[numPoints];
	uint64_t* tiles_touched_host = new uint64_t[numPoints];

	// 从 GPU 复制数据到 CPU
	//cudaMemcpy(tiles_touched_host, geomState.tiles_touched, numPoints * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(tiles_touched_host, geomState.tiles_touched, numPoints * sizeof(uint64_t), cudaMemcpyDeviceToHost);

	// 打印 point_offsets_host 中的所有值
	for (size_t i = 0; i < numPoints; ++i) {
		//printf("geomState.tiles_touched[%zu]: %llu\n", i, geomState.tiles_touched[i]);
		printf("geomState.tiles_touched[%zu]: %llu\n", i, tiles_touched_host[i]);		
	}

	// 释放 CPU 上的内存
	delete[] tiles_touched_host;
	*/
	/*
	for (int i = 0; i < P; i++) {
		//printf("InclusiveSum geomState tiles_touched:%u\n", geomState.tiles_touched[i]);
		//printf("InclusiveSum geomState depths:%f\n", geomState.depths[i]);
	}
	*/
	//Really fill geomState(partially)
	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		geomState.clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,
		prefiltered
	), debug)

	printf("preprocess after\n");
	/*
	printf("InclusiveSum geomState depths:%ld\n", geomState.depths);
	printf("InclusiveSum geomState point_offsets:%ld\n", geomState.point_offsets);
	printf("InclusiveSum geomState tiles_touched:%ld\n", geomState.tiles_touched);
	printf("InclusiveSum geomState scanning_space:%ld\n", geomState.scanning_space);
	printf("InclusiveSum geomState clamped:%ld\n", geomState.clamped);
	printf("InclusiveSum geomState internal_radii:%ld\n", geomState.internal_radii);	
	printf("InclusiveSum geomState means2D:%ld\n", geomState.means2D);
	printf("InclusiveSum geomState cov3D:%ld\n", geomState.cov3D);	
	printf("InclusiveSum geomState conic_opacity:%ld\n", geomState.conic_opacity);
	printf("InclusiveSum geomState rgb:%ld\n", geomState.rgb);
	*/
	// 在 CPU 上分配内存以存储 point_offsets 的副本
	//int numPoints = P;
	/*
	uint32_t* point_offsets_host = new uint32_t[numPoints];

	// 从 GPU 复制数据到 CPU
	cudaMemcpy(point_offsets_host, geomState.point_offsets, numPoints * sizeof(uint32_t), cudaMemcpyDeviceToHost);

	// 打印 point_offsets_host 中的所有值
	for (size_t i = 0; i < numPoints; ++i) {
		printf("geomState.point_offsets[%zu]: %u\n", i, point_offsets_host[i]);
	}

	// 释放 CPU 上的内存
	delete[] point_offsets_host;
	*/
	/*
	//uint32_t* tiles_touched_host1 = new uint32_t[numPoints];
	uint64_t* tiles_touched_host1 = new uint64_t[numPoints];

	// 从 GPU 复制数据到 CPU
	//cudaMemcpy(tiles_touched_host1, geomState.tiles_touched, numPoints * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(tiles_touched_host1, geomState.tiles_touched, numPoints * sizeof(uint64_t), cudaMemcpyDeviceToHost);

	// 打印 point_offsets_host 中的所有值
	for (size_t i = 0; i < numPoints; ++i) {
		//printf("geomState.tiles_touched_host1[%zu]: %u\n", i, tiles_touched_host1[i]);
		//printf("geomState.tiles_touched[%zu]: %llu\n", i, geomState.tiles_touched[i]);
		printf("geomState.tiles_touched_host1[%zu]: %llu\n", i, tiles_touched_host1[i]);
	}

	// 释放 CPU 上的内存
	delete[] tiles_touched_host1;
	*/
	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	printf("InclusiveSum geomState scan_size:%zu\n", geomState.scan_size);
    /*
	//fill geomState.point_offsets
	for (int i = 0; i < 1; i++) {
		printf("InclusiveSum geomState tiles_touched:%u\n", geomState.tiles_touched[i]);
		printf("InclusiveSum geomState depths:%f\n", geomState.depths[i]);
	}
	*/
	//overflow!! uint32_t 0到4294967295 uint64_t: 0 - 2^64 2^64 = 18446744073709551615 not fixed!
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)
	/*
	for (int i = 0; i < P; i++) {
		printf("InclusiveSum geomState point_offsets:%u\n", geomState.point_offsets[i]);
	}
	*/
	/*
	//uint32_t* point_offsets_host = new uint32_t[numPoints];
	uint64_t* point_offsets_host = new uint64_t[numPoints];
	// 从 GPU 复制数据到 CPU
	//cudaMemcpy(point_offsets_host, geomState.point_offsets, numPoints * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(point_offsets_host, geomState.point_offsets, numPoints * sizeof(uint64_t), cudaMemcpyDeviceToHost);

	// 打印 point_offsets_host 中的所有值
	for (size_t i = 0; i < numPoints; ++i) {
		//printf("geomState.point_offsets[%zu]: %u\n", i, point_offsets_host[i]);
		printf("geomState.point_offsets[%zu]: %llu\n", i, point_offsets_host[i]);
	}


	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	//printf("cudaMemcpy last point_offsets:%ld \n", *(geomState.point_offsets + P - 1));
	//int num_rendered = point_offsets_host[P - 1];
	// 释放 CPU 上的内存
	delete[] point_offsets_host;
	*/
	
	int num_rendered;
	//CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost), debug);
	//CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(uint64_t), cudaMemcpyDeviceToHost), debug);
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);	
    	printf("before required BinningState ... num_rendered %ld\n", num_rendered); //not fixed!

	printf("required BinningState \n");
	size_t binning_chunk_size = required<BinningState>(num_rendered);
	printf("before binningBuffer... binning_chunk_size size %zu\n", binning_chunk_size);

	char* binning_chunkptr = binningBuffer(binning_chunk_size); //so large! numel: integer multiplication overflow
	printf("fromChunk binningState\n");
	//return num_rendered; //bad!

	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

    	//return num_rendered; //bad!
	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding duplicated Gaussian indices to be sorted
	printf("duplicateWithKeys \n");
	duplicateWithKeys <<<(P + 255) / 256, 256 >>> (
	//hipLaunchKernelGGL(duplicateWithKeys, dim3((P + 255) / 256), dim3(256), 0, 0,
	//duplicateWithKeys <<< dim3((P + 255) / 256), dim3(256), 0, hipStreamDefault >>> (
		P,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		tile_grid);
	CHECK_CUDA(, debug)
    	//return num_rendered; //good!

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	printf("SortPairs \n");
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug)

    	//return num_rendered; //good!
    	printf("cudaMemset \n");
	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);
    	//return num_rendered; //good!

	// Identify start and end of per-tile workloads in sorted list
	printf("identifyTileRanges num_rendered:%ld, tile_gridx:%d, tile_gridy:%d\n", num_rendered, tile_grid.x, tile_grid.y);
	//really fill imgState?

	if (num_rendered > 0) {
		printf("num_rendered > 0\n");
	    identifyTileRanges <<<(num_rendered + 255) / 256, 256 >>> (
		//hipLaunchKernelGGL(identifyTileRanges, dim3((num_rendered + 255) / 256), dim3(256), 0, 0,
		//identifyTileRanges <<< dim3((num_rendered + 255) / 256), dim3(256), 0, hipStreamDefault >>> (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	}
	printf("identifyTileRanges finish!\n");
	CHECK_CUDA(, debug);
	printf("identifyTileRanges synced!\n");
    	//return num_rendered; //bad!
	//really fill imgState?
	//Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	printf("FORWARD::render blockx:%d blocky:%d\n", block.x, block.y);
	CHECK_CUDA(FORWARD::render(
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		geomState.means2D,
		feature_ptr,
		geomState.depths,
		geomState.conic_opacity,
		out_alpha,
		imgState.n_contrib,
		background,
		out_color,
		out_depth), debug);

	return num_rendered;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
	const int P, int D, int M, int R,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* alphas,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* campos,
	const float tan_fovx, float tan_fovy,
	const int* radii,
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	const float* dL_dpix,
	const float* dL_dpix_depth,
	const float* dL_dalphas,
	float* dL_dmean2D,
	float* dL_dconic,
	float* dL_dopacity,
	float* dL_dcolor,
	float* dL_ddepth,
	float* dL_dmean3D,
	float* dL_dcov3D,
	float* dL_dsh,
	float* dL_dscale,
	float* dL_drot,
	bool debug)
{
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	const float* depth_ptr = geomState.depths;
	CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		background,
		geomState.means2D,
		geomState.conic_opacity,
		color_ptr,
		depth_ptr,
		alphas,
		imgState.n_contrib,
		dL_dpix,
		dL_dpix_depth,
		dL_dalphas,
		(float3*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor,
		dL_ddepth), debug)

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		shs,
		geomState.clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D,
		dL_dconic,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_ddepth,
		dL_dcov3D,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot), debug)
}
