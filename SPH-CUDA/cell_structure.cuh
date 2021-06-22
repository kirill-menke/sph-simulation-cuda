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

__global__ void assign_to_cells(Particle* particles, int* cell_list, int* particle_list, int N, float3 cells_dim, float3 min_box_bound, float h_inv) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid < N) {
		int3 cell_idx = floor((particles[tid].pos - min_box_bound) * h_inv);
		int flat_cell_idx = cell_idx.x + cells_dim.x * cell_idx.y + cells_dim.x * cells_dim.y * cell_idx.z;
		particle_list[tid] = atomicExch(&cell_list[flat_cell_idx], tid);
	}
}