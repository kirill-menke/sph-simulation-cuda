#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "helper_structs.h"
#include "helper_math.h"

#include "src/MarchingCubes/CudaGrid.h"

__global__ void copy_triangles(float* vertexArray, float3* vertices, int N, float3 min_box_bound, CudaGrid grid) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid < N) {
        float3 gridPos = vertices[tid] / grid.voxelSpacing;
        //if (tid%777)
		//	printf("%f\n", gridPos.x);
        uint3 idx = make_uint3(gridPos.x, gridPos.y, gridPos.z);

		vertexArray[tid*9 + 0] = vertices[tid].x + min_box_bound.x;
        vertexArray[tid*9 + 1] = vertices[tid].y + min_box_bound.y;
        vertexArray[tid*9 + 2] = vertices[tid].z + min_box_bound.z;

		// rgb color
		vertexArray[tid*9 + 3] = 0.2;
        vertexArray[tid*9 + 4] = 0.2;
        vertexArray[tid*9 + 5] = 0.8;

        vertexArray[tid*9 + 6] = grid(idx).normal.x;
        vertexArray[tid*9 + 7] = grid(idx).normal.y;
        vertexArray[tid*9 + 8] = grid(idx).normal.z;
	}
}

__device__ float get_density(float3 pos, Particle* particles, int* cell_list, int* particle_list, float* density_buffer,
	float3 cell_dims, float3 min_box_bound, int immovable_particle_num, float h, float h2, float h_inv, float const_poly6, float mass, float p0) {

	int3 cell_idx = floor((pos - min_box_bound) * h_inv);
	float density = 0;

	for (int x = -1; x <= 1; x++) {
		for (int y = -1; y <= 1; y++) {
			for (int z = -1; z <= 1; z++) {
				int3 neighbor_cell_idx = cell_idx + make_int3(x, y, z);
				if (neighbor_cell_idx.x < 0 || neighbor_cell_idx.y < 0 || neighbor_cell_idx.z < 0 || neighbor_cell_idx.x >= cell_dims.x || neighbor_cell_idx.y >= cell_dims.y || neighbor_cell_idx.z >= cell_dims.z) {
					continue;
				}
				int neighbor_flat_idx = neighbor_cell_idx.x + neighbor_cell_idx.y * cell_dims.x + neighbor_cell_idx.z * cell_dims.x * cell_dims.y;

				int neighbor_particle_idx = cell_list[neighbor_flat_idx];
				while (neighbor_particle_idx != -1) {

					Particle& particleB = particles[neighbor_particle_idx];
					
					// Evaluate density
					float3 diff = pos - particleB.pos;
					float r2 = dot(diff, diff);

					if (r2 < h2 && neighbor_particle_idx > immovable_particle_num) {
						density += mass * const_poly6 * powf(h2 - r2, 3);
					}

					// Follow linked list
					neighbor_particle_idx = particle_list[neighbor_particle_idx];
				}
			}
		}
	}

	// Density can't be lower than reference density to avoid negative pressure
	//density = fmaxf(p0, density);
	return density;
}

__global__ void fill_grid(CudaGrid grid, Particle* particles, int N, int* cell_list, int* particle_list, float* density_buffer,
	float3 cell_dims, float3 min_box_bound, int immovable_particle_num, float h, float h2, float h_inv, float const_poly6, float mass, float p0) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid < N) {
		// (idx.z * gridSize.y + idx.y) * gridSize.x + idx.x
		int z = tid / (grid.gridSize.x * grid.gridSize.y);
		int y = (tid % (grid.gridSize.x * grid.gridSize.y)) / grid.gridSize.x;
		int x = tid % (grid.gridSize.x);
		uint3 gridPos = make_uint3(x, y, z);
		float3 fpos = make_float3(x, y, z);
		float3 pos = min_box_bound + fpos * grid.voxelSpacing;
		float density;
		float b = 1;
		if (x <= b || y <= b || z <= b || x >= grid.gridSize.x-b-1 || y >= grid.gridSize.y-b-1 || z >= grid.gridSize.z-b-1) {
			density = 0;
		} else {
			density = get_density(pos, particles, cell_list, particle_list, density_buffer, cell_dims, min_box_bound, immovable_particle_num, h, h2, h_inv, const_poly6, mass, p0);
		}
		grid(gridPos).value = density;
	}
}

__global__ void update_grid_normals(CudaGrid grid, Particle* particles, int N, int* cell_list, int* particle_list, float* density_buffer,
	float3 cell_dims, float3 min_box_bound, int numParticles, float h, float h2, float h_inv, float const_poly6, float mass, float p0) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid < N) {
		// (idx.z * gridSize.y + idx.y) * gridSize.x + idx.x
		int z = tid / (grid.gridSize.x * grid.gridSize.y);
		int y = (tid % (grid.gridSize.x * grid.gridSize.y)) / grid.gridSize.x;
		int x = tid % (grid.gridSize.x);

        if (x >= grid.gridSize.x - 1 || y >= grid.gridSize.y - 1 || z >= grid.gridSize.z - 1)
            return;
		uint3 gridPos = make_uint3(x, y, z);
		float3 fpos = make_float3(x, y, z);
		float3 pos = min_box_bound + fpos * grid.voxelSpacing;
		float normalX = grid(gridPos).value - grid(gridPos.x + 1, gridPos.y, gridPos.z).value;
        float normalY = grid(gridPos).value - grid(gridPos.x, gridPos.y + 1, gridPos.z).value;
        float normalZ = grid(gridPos).value - grid(gridPos.x, gridPos.y, gridPos.z + 1).value;
        grid(gridPos).normal = make_float3(normalX, normalY, normalZ);
	}
}