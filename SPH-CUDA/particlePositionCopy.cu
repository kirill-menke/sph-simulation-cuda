#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "helper_structs.h"
#include "helper_math.h"

__global__ void copy_particle_positions(float* translations, Particle* particles, int N, int immovable_particle_num, int dam_particle_num) {
	
	#if defined(RENDERWALLS)
		// Rendering wall particles
		int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

		if (tid >= 0 && tid < immovable_particle_num) {
			translations[(tid) * 3 + 0] = particles[tid].pos.x;
			translations[(tid) * 3 + 1] = particles[tid].pos.y;
			translations[(tid) * 3 + 2] = particles[tid].pos.z;
		}

		tid = (blockIdx.x * blockDim.x) + threadIdx.x + immovable_particle_num;
		if (tid >= immovable_particle_num && tid < N) {
			translations[(tid) * 3 + 0] = particles[tid].pos.x;
			translations[(tid) * 3 + 1] = particles[tid].pos.y;
			translations[(tid) * 3 + 2] = particles[tid].pos.z;
		}

	#else

		int tid = (blockIdx.x * blockDim.x) + threadIdx.x + immovable_particle_num - dam_particle_num;
		if (tid >= immovable_particle_num - dam_particle_num && tid < N) {
			translations[(tid - immovable_particle_num + dam_particle_num) * 3 + 0] = particles[tid].pos.x;
			translations[(tid - immovable_particle_num + dam_particle_num) * 3 + 1] = particles[tid].pos.y;
			translations[(tid - immovable_particle_num + dam_particle_num) * 3 + 2] = particles[tid].pos.z;
		}

	#endif
}