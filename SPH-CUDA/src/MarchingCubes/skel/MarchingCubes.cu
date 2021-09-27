#include "../MarchingCubes.h"
#include "../CudaGrid.h"
#include "../cutil_math.h"
#include "../Tables.h"
#include "../cutil.h"
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <iostream>
#include <device_atomic_functions.h>
#include "cuda_runtime_api.h"

__device__ void MarchingCubes::MarchingCubesData::addTriangle(const Triangle &t) {
	uint slot = 0;

	// get slot & increment tri counter	
	slot = (uint) atomicAdd(d_numTriangles, 1);

	if (slot >= maxNumTriangles) {
		printf("ERROR: #triangles: %i, max #triangles: %i\n", slot, maxNumTriangles);
		return;
	}

	// store t in slot
	d_triangles[slot] = t;

}

__device__ MarchingCubes::Vertex MarchingCubes::vertexInterp(float isolevel, const float3& p1, const float3& p2, float d1, float d2) {
	MarchingCubes::Vertex r1; r1.position = p1;
	MarchingCubes::Vertex r2; r2.position = p2;
	float mu = -1.f;
	float c = isolevel;
	// compute weight mu using c = (1 - mu) * d1 + mu * d2
	mu = (c - d1) / (d2 - d1);

	MarchingCubes::Vertex res;
	// compute res.position
	res.position = lerp(r1.position, r2.position, mu);
	return res;
}

__global__ void extractTrianglesKernel(CudaGrid grid, float isoValue, MarchingCubes::MarchingCubesData data) {
	const uint3 idx = (dim3)blockDim * blockIdx + threadIdx;
	//only extract no boundary
	if (idx.x == 0 || idx.y == 0 || idx.z == 0) {
		return;
	}
	if (idx.x >= grid.gridSize.x - 2 || idx.y >= grid.gridSize.y - 2 || idx.z >= grid.gridSize.z - 2) {
		return;
	}
	
	const uint3 idx0 = idx + make_uint3(0, 1, 0);
	const uint3 idx1 = idx + make_uint3(1, 1, 0);
	const uint3 idx2 = idx + make_uint3(1, 0, 0);
	const uint3 idx3 = idx + make_uint3(0, 0, 0);
	const uint3 idx4 = idx + make_uint3(0, 1, 1);
	const uint3 idx5 = idx + make_uint3(1, 1, 1);
	const uint3 idx6 = idx + make_uint3(1, 0, 1);
	const uint3 idx7 = idx + make_uint3(0, 0, 1);

	const float dist0 = grid(idx0).value;
	const float dist1 = grid(idx1).value;
	const float dist2 = grid(idx2).value;
	const float dist3 = grid(idx3).value;
	const float dist4 = grid(idx4).value;
	const float dist5 = grid(idx5).value;
	const float dist6 = grid(idx6).value;
	const float dist7 = grid(idx7).value;

	float3 pos0 = make_float3(idx0) * grid.voxelSpacing; 
	float3 pos1 = make_float3(idx1) * grid.voxelSpacing; 
	float3 pos2 = make_float3(idx2) * grid.voxelSpacing;
	float3 pos3 = make_float3(idx3) * grid.voxelSpacing;
	float3 pos4 = make_float3(idx4) * grid.voxelSpacing;
	float3 pos5 = make_float3(idx5) * grid.voxelSpacing;
	float3 pos6 = make_float3(idx6) * grid.voxelSpacing;
	float3 pos7 = make_float3(idx7) * grid.voxelSpacing;
	

	unsigned int cubeindex = 0;
	// compute cubeindex & voxel positions
	// cubeindex is an 8 bit index, set respective bit to generate cubeindex

	if (dist0 < isoValue) {
		cubeindex += 1;
	}
	if (dist1 < isoValue) {
		cubeindex += 2;
	}
	if (dist2 < isoValue) {
		cubeindex += 4;
	}
	if (dist3 < isoValue) {
		cubeindex += 8;
	}
	if (dist4 < isoValue) {
		cubeindex += 16;
	}
	if (dist5 < isoValue) {
		cubeindex += 32;
	}
	if (dist6 < isoValue) {
		cubeindex += 64;
	}
	if (dist7 < isoValue) {
		cubeindex += 128;
	}


	MarchingCubes::Vertex vertlist[12];
	// generate vertexlist
	// edgeTable returns a 12 bit number, each bit corresponds to 1 of 12 edges
	if (edgeTable[cubeindex] & 1) {
		vertlist[0] = MarchingCubes::vertexInterp(isoValue, pos0, pos1, dist0, dist1);
	}
	if (edgeTable[cubeindex] & 2) {
		vertlist[1] = MarchingCubes::vertexInterp(isoValue, pos1, pos2, dist1, dist2);
	}
	if (edgeTable[cubeindex] & 4) {
		vertlist[2] = MarchingCubes::vertexInterp(isoValue, pos2, pos3, dist2, dist3);
	}
	if (edgeTable[cubeindex] & 8) {
		vertlist[3] = MarchingCubes::vertexInterp(isoValue, pos3, pos0, dist3, dist0);
	}
	if (edgeTable[cubeindex] & 16) {
		vertlist[4] = MarchingCubes::vertexInterp(isoValue, pos4, pos5, dist4, dist5);
	}
	if (edgeTable[cubeindex] & 32) {
		vertlist[5] = MarchingCubes::vertexInterp(isoValue, pos5, pos6, dist5, dist6);
	}
	if (edgeTable[cubeindex] & 64) {
		vertlist[6] = MarchingCubes::vertexInterp(isoValue, pos6, pos7, dist6, dist7);
	}
	if (edgeTable[cubeindex] & 128) {
		vertlist[7] = MarchingCubes::vertexInterp(isoValue, pos7, pos4, dist7, dist4);
	}
	if (edgeTable[cubeindex] & 256) {
		vertlist[8] = MarchingCubes::vertexInterp(isoValue, pos0, pos4, dist0, dist4);
	}
	if (edgeTable[cubeindex] & 512) {
		vertlist[9] = MarchingCubes::vertexInterp(isoValue, pos1, pos5, dist1, dist5);
	}
	if (edgeTable[cubeindex] & 1024) {
		vertlist[10] = MarchingCubes::vertexInterp(isoValue, pos2, pos6, dist2, dist6);
	}
	if (edgeTable[cubeindex] & 2048) {
		vertlist[11] = MarchingCubes::vertexInterp(isoValue, pos3, pos7, dist3, dist7);
	}


	
	for (int i = 0; triTable[cubeindex][i] != -1; i += 3) {
		MarchingCubes::Triangle t;
		// compute tri
		t.v[0] = vertlist[triTable[cubeindex][i]];
		t.v[1] = vertlist[triTable[cubeindex][i+1]];
		t.v[2] = vertlist[triTable[cubeindex][i+2]];

		data.addTriangle(t);
	}
}

void MarchingCubes::extractTrianglesGPU(const CudaGrid &grid, float isoValue) {
	const dim3 blockDim(8, 8, 8);
	const dim3 gridDim((grid.gridSize.x + blockDim.x - 1) / blockDim.x, (grid.gridSize.y + blockDim.y - 1) / blockDim.y, (grid.gridSize.z + blockDim.z - 1) / blockDim.z);

	extractTrianglesKernel <<< gridDim, blockDim >>> (grid, isoValue, marchingCubesData);
	cudaDeviceSynchronize();

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
#endif
}