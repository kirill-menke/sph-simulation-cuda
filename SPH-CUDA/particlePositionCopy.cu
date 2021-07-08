#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "helper_structs.h"
#include "helper_math.h"

__global__ void copy_particle_positions(float* translations, Particle* particles, int N) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid < N) {
		translations[tid*3 + 0] = particles[tid].pos.x;
        translations[tid*3 + 1] = particles[tid].pos.y;
        translations[tid*3 + 2] = particles[tid].pos.z;
	}
}