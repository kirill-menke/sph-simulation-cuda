#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "helper_structs.h"
#include "helper_math.h"

#include "cell_structure.cuh"

__global__ void 
calculate_density(Particle* particles, int* cell_list, int* particle_list, float* density_buffer,
	float3 cell_dims, float3 min_box_bound, int N, int immovable_particle_num, float h, float h2, float h_inv, float const_poly6, float mass, float p0) {

	int tid = (blockIdx.x * blockDim.x) + threadIdx.x + immovable_particle_num;

	if (tid >= immovable_particle_num && tid < N) {
		Particle& particleA = particles[tid];
		int3 cell_idx = calculate_cell_idx(particleA, min_box_bound, cell_dims, h_inv, tid);
		if (cell_idx.x == -1) {
			return;
		}
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
						float3 diff = particleA.pos - particleB.pos;
						float r2 = dot(diff, diff);

						if (r2 < h2) {
							density += mass * const_poly6 * powf(h2 - r2, 3);
						}

						// Follow linked list
						neighbor_particle_idx = particle_list[neighbor_particle_idx];
					}
				}
			}
		}

		// Density can't be lower than reference density to avoid negative pressure
		density = fmaxf(p0, density);

		density_buffer[tid] = density;
	}
}