#pragma once
#include <vector_types.h>
#include "vector_functions.h"
#include "cutil_math.h"
//#define SKELETON

class CudaGrid {
public:
	CudaGrid(uint3 gridSize);
	void free();

	static CudaGrid Sphere(uint3 gridSize);

	struct Voxel {
		float value;
		float3 normal;
	};

	__device__ Voxel &operator()(const uint3 &idx) {
		return d_memory[getlinearIndex(idx)];
	}

	__device__ const Voxel &operator()(const uint3 &idx) const {
		return d_memory[getlinearIndex(idx)];
	}

	__device__ Voxel &operator()(unsigned int x, unsigned int y, unsigned int z) {
		CudaGrid &grid = *this;
		return grid(make_uint3(x, y, z));
	}
	__device__ const Voxel &operator()(unsigned int x, unsigned int y, unsigned int z) const {
		const CudaGrid &grid = *this;
		return grid(make_uint3(x, y, z));
	}

	__device__ unsigned int getlinearIndex(const uint3 &idx) const { 
		return (idx.z * gridSize.y + idx.y) * gridSize.x + idx.x; 
	}

	__device__ float interpolateScalarValue(const float3 &idx) const;

	const uint3 gridSize;
	const float3 voxelSpacing;

protected:
	Voxel *d_memory = nullptr;
};
