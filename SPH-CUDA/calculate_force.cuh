#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "helper_structs.h"
#include "helper_math.h"

#include "cell_structure.cuh"

__global__ void set_dam_force(Particle* particle, float3* force_buffer, int N, int immovable_particle_num, int dam_particle_num) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x + immovable_particle_num - dam_particle_num;

	float3 up_force = make_float3(0, 20, 0);
	if (tid >= immovable_particle_num - dam_particle_num && tid < immovable_particle_num) {
		force_buffer[tid] = up_force;
	}
}

__global__ void
calculate_force(Particle* particles, int* cell_list, int* particle_list, float3* force_buffer, float* density_buffer, float3 cell_dims, float3 min_box_bound, 
	int N, int immovable_particle_num, float h, float h_inv, float const_spiky, float const_visc, float const_surf, const float mass, float k, float e, float p0, float s, float3 g) {

	int tid = (blockIdx.x * blockDim.x) + threadIdx.x + immovable_particle_num;

	if (tid >= immovable_particle_num && tid < N) {
		Particle& particleA = particles[tid];
		int3 cell_idx = calculate_cell_idx(particleA, min_box_bound, cell_dims, h_inv, tid);
		if (cell_idx.x == -1) {
			return;
		}

		float3 f_pressure = make_float3(0, 0, 0);
		float3 f_viscosity = make_float3(0, 0, 0);
		float3 f_surface = make_float3(0, 0, 0);

		float densityA = density_buffer[tid];
		float pressureA = (k * p0 / 7) * (pow(((densityA) / p0), 7) - 1);
		
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

						float3 diff = particleA.pos - particleB.pos;
						float r2 = dot(diff, diff);
						float r = sqrtf(r2);
						float r3 = powf(r, 3.);

						// Exclude current particle (r = 0) from force calculation
						if (r > 0 && r < h)
						{
							// Normalize distance between particles
							float3 r_norm = diff / r;

							// Pressure forces
							float3 w_press = const_spiky * powf(h - r, 2) * r_norm;
							float densityB = density_buffer[neighbor_particle_idx];
							float pressureB = fmaxf((k * p0 / 7) * (pow(((densityB) / p0), 7) - 1), 0);
							f_pressure += mass * (pressureA / (densityA * densityA) + pressureB / (densityB * densityB)) * w_press;


							// Viscosity forces
							float w_vis = -const_spiky * (h - r);
							float3 v_diff = particleB.vel - particleA.vel;
							f_viscosity += mass * (v_diff / densityB) * w_vis;


							//// Surface tension
							//float w_surf = 0;
							//float q = (r / h) * 2;
							//if (0 < q <= 1)
							//	w_surf = 2/3 - powf(q, 2) + 0.5 * powf(q, 3);
							//else if (1 < q < 2)
							//	w_surf = 1/6 * powf(2 - q, 3);

							//f_surface += mass * const_surf * w_surf * diff;

						}

						// Follow linked list
						neighbor_particle_idx = particle_list[neighbor_particle_idx];
					}
				}
			}
		}


		f_pressure *= -1;
		f_surface *= - s / mass;
		f_viscosity *= e;


		force_buffer[tid] = (f_pressure + f_viscosity) / densityA + g;
	}
}