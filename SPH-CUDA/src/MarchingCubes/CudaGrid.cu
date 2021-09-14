#include "CudaGrid.h"
#include "cutil_math.h"
#include "cutil.h"
#include <device_launch_parameters.h>

__device__ float CudaGrid::interpolateScalarValue(const float3 &idxIn) const {
	const float3 idx = clamp(idxIn, 0, 1);

	const uint3 cellIdx = make_uint3(idx.x * voxelSpacing.x, idx.y * voxelSpacing.y, idx.z * voxelSpacing.z);
	const uint3 cellIdx_max = min(cellIdx + 1, gridSize);
	const float3 cellPos = make_float3(cellIdx) / voxelSpacing;
	const float3 uvw = (idx - cellPos) * voxelSpacing;

	const CudaGrid &g = *this;

	float s000 = g(cellIdx).value;
	float s001 = g(make_uint3(cellIdx.x, cellIdx.y, cellIdx_max.z)).value;
	float s010 = g(make_uint3(cellIdx.x, cellIdx_max.y, cellIdx.z)).value;
	float s011 = g(make_uint3(cellIdx.x, cellIdx_max.y, cellIdx_max.z)).value;
	float s100 = g(make_uint3(cellIdx_max.x, cellIdx.y, cellIdx.z)).value;
	float s101 = g(make_uint3(cellIdx_max.x, cellIdx.y, cellIdx_max.z)).value;
	float s110 = g(make_uint3(cellIdx_max.x, cellIdx_max.y, cellIdx.z)).value;
	float s111 = g(cellIdx_max).value;

	return lerp(lerp(lerp(s000, s001, uvw.z), lerp(s010, s011, uvw.z), uvw.y),
		lerp(lerp(s100, s101, uvw.z), lerp(s110, s111, uvw.z), uvw.y), uvw.x);
}

__global__ void sphereKernel(CudaGrid grid) {
	const uint3 idx = (dim3)blockDim * blockIdx + threadIdx;

	if (idx.x >= grid.gridSize.x || idx.y >= grid.gridSize.y || idx.z >= grid.gridSize.z) {
		return;
	}

	const float3 pos = make_float3(idx);
	const float3 center = make_float3(grid.gridSize) / 2.f;
	const float radius = grid.gridSize.x / 4.f;

	grid(idx).value = length(pos - center) - radius;
}

CudaGrid CudaGrid::Sphere(uint3 gridSize, float3 voxelSpacing) {
	CudaGrid grid(gridSize, voxelSpacing);

	const dim3 blockDim(8, 8, 8);
	const dim3 gridDim((gridSize.x + blockDim.x - 1) / blockDim.x, (gridSize.y + blockDim.y - 1) / blockDim.y, (gridSize.z + blockDim.z - 1) / blockDim.z);

	sphereKernel << <gridDim, blockDim >> > (grid);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
#endif

	return grid;
}