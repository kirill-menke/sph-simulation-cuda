#include "../MarchingCubes.h"
#include "cuda_runtime.h"
#include "../cutil.h"
#include <iostream>

MarchingCubes::MarchingCubes(unsigned int maxNumTriangles) {
	marchingCubesData.maxNumTriangles = maxNumTriangles;	

	const size_t numBytes = maxNumTriangles * sizeof(Triangle);	

	cutilSafeCall(cudaMalloc(&marchingCubesData.d_triangles, numBytes));
	cutilSafeCall(cudaMalloc(&marchingCubesData.d_numTriangles, sizeof(unsigned int)));
}

std::vector<MarchingCubes::Triangle> MarchingCubes::extractTriangles(const CudaGrid &grid, float isoValue) {
	unsigned int numTriangles = 0;

	// init marchingCubesData.d_numTriangles
	cudaMemset(marchingCubesData.d_numTriangles, 0, sizeof(unsigned int));

	extractTrianglesGPU(grid, isoValue);

	// get num of extracted triangles
	cutilSafeCall(cudaMemcpy(&numTriangles, marchingCubesData.d_numTriangles, sizeof(int), cudaMemcpyDeviceToHost));
	std::vector<Triangle> triangles(numTriangles);

	// copy result to cpu
	cutilSafeCall(cudaMemcpy(triangles.data(), marchingCubesData.d_triangles, numTriangles * sizeof(Triangle), cudaMemcpyDeviceToHost));

	return triangles;
}