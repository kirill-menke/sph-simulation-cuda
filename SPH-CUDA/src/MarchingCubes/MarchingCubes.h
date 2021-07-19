#pragma once
#include "CudaGrid.h"
#include <vector_types.h>
#include <vector>

class MarchingCubes {
public:
	MarchingCubes(unsigned int maxNumTriangles);

	struct Vertex {
		float3 position;
	};

	struct Triangle {
		Vertex v[3];
	};

	struct MarchingCubesData {
		__device__ void addTriangle(const Triangle &t);

		unsigned int maxNumTriangles = 0;			// maximum number of triangles
		unsigned int *d_numTriangles = nullptr;		// current number of extracted triangles
		
		Triangle *d_triangles = nullptr;			// buffer to store the triangles
	};

	std::vector<Triangle> extractTriangles(const CudaGrid &grid, float isoValue);

	__device__ static Vertex vertexInterp(float isolevel, const float3& p1, const float3& p2, float d1, float d2);

	void extractTrianglesGPU(const CudaGrid &grid, float isoValue);	
	
	MarchingCubesData marchingCubesData;
};