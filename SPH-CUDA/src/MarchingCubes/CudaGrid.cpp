#include "CudaGrid.h"
#include "cuda_runtime.h"
#include "cutil.h"
#include <iostream>

CudaGrid::CudaGrid(uint3 gridSize, float3 voxelSpacing) : gridSize(gridSize), voxelSpacing(voxelSpacing) {
	const size_t numBytes = gridSize.x * gridSize.y * gridSize.z * sizeof(Voxel);

	cutilSafeCall(cudaMalloc(&d_memory, numBytes));
}

void CudaGrid::free() {
	cutilSafeCall(cudaFree(d_memory));
}