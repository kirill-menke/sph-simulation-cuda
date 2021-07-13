#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "helper_structs.h"
#include "helper_math.h"

__global__ void reset_cell_list(int* cell_list, size_t N) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (tid < N) {
		cell_list[tid] = -1;
	}
}

__device__ int3 calculate_cell_idx(Particle particle, float3 min_box_bound, float3 cell_dims, float h_inv, size_t tid) {

	int3 cell_idx = floor((particle.pos - (min_box_bound)) * h_inv);
	if (cell_idx.x < 0 || cell_idx.y < 0 || cell_idx.z < 0 || cell_idx.x >= cell_dims.x || cell_idx.y >= cell_dims.y || cell_idx.z >= cell_dims.z) {
		// Avoid illegal memory access
		return make_int3(-1, -1, -1);
	}

	return cell_idx;

}

__global__ void assign_to_cells(Particle* particles, int* cell_list, int* particle_list, int N, int immovable_particle_num, float3 cell_dims, float3 min_box_bound, float h_inv) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	// Assignment for immovable particles
	if (tid >= 0 && tid < immovable_particle_num) {
		int3 cell_idx = calculate_cell_idx(particles[tid], min_box_bound, cell_dims, h_inv, tid);
		if (cell_idx.x != -1) {
			int flat_cell_idx = cell_idx.x + cell_dims.x * cell_idx.y + cell_dims.x * cell_dims.y * cell_idx.z;
			particle_list[tid] = atomicExch(&cell_list[flat_cell_idx], tid);
		}
	}

	// Assignment for moveable particles
	tid += immovable_particle_num;
	if (tid >= immovable_particle_num && tid < N ) {
		int3 cell_idx = calculate_cell_idx(particles[tid], min_box_bound, cell_dims, h_inv, tid);
		if (cell_idx.x != -1) {
			int flat_cell_idx = cell_idx.x + cell_dims.x * cell_idx.y + cell_dims.x * cell_dims.y * cell_idx.z;
			particle_list[tid] = atomicExch(&cell_list[flat_cell_idx], tid);
		}
	}
}
