#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "helper_structs.h"
#include "helper_math.h"

__global__ void
integrate_symplectic_euler(Particle* particles, float3* force_buffer, double delta_time, size_t N, float3 min_box_bound, float3 max_box_bound, float damping) {

	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (tid < N) {
		Particle& particle = particles[tid];
		
		particle.vel += delta_time * force_buffer[tid];
		particle.pos += delta_time * particle.vel;

		// Boundary condition: Dampen and reverse velocity vector
        if (particle.pos.x < min_box_bound.x) {
            particle.pos.x = min_box_bound.x;
            particle.vel.x = -particle.vel.x * damping;
        }
        else if (particle.pos.x > max_box_bound.x) {
            particle.pos.x = max_box_bound.x;
            particle.vel.x = -particle.vel.x * damping;
        }

        if (particle.pos.y < min_box_bound.y) {
            particle.pos.y = min_box_bound.y;
            particle.vel.y = -particle.vel.y * damping;
        }
        else if (particle.pos.y > max_box_bound.z - min_box_bound.z) {
            particle.pos.y = max_box_bound.z - min_box_bound.z;
            particle.vel.y = -particle.vel.y * damping;
        }

        if (particle.pos.z < min_box_bound.z) {
            particle.pos.z = min_box_bound.z;
            particle.vel.z = -particle.vel.z * damping;
        }
        else if (particle.pos.z > max_box_bound.z) {
            particle.pos.z = max_box_bound.z;
            particle.vel.z = -particle.vel.z * damping;
        }
	}
}