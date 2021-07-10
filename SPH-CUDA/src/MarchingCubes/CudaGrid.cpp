#include "CudaGrid.h"
#include "cuda_runtime.h"
#include "cutil.h"
#include <iostream>

CudaGrid::CudaGrid(uint3 gridSize) : gridSize(gridSize), voxelSpacing(make_float3(1.f / (gridSize.x - 1.f), 1.f / (gridSize.y - 1.f), 1.f / (gridSize.z - 1.f))){
	const size_t numBytes = gridSize.x * gridSize.y * gridSize.z * sizeof(Voxel);

	cutilSafeCall(cudaMalloc(&d_memory, numBytes));
}

void CudaGrid::free() {
	cutilSafeCall(cudaFree(d_memory));
}