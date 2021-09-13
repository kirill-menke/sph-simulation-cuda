#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "helper_structs.h"
#include "helper_math.h"


__global__ void
integrate_symplectic_euler(Particle* particles, float3* force_buffer, float delta_time, int N, int immovable_particle_num, int dam_particle_num, float3 min_box_bound, float3 max_box_bound, float damping) {

	int tid = (blockIdx.x * blockDim.x) + threadIdx.x + immovable_particle_num - dam_particle_num;

    /* Integrate position and velocity only for movable particles */
	if (tid >= immovable_particle_num - dam_particle_num && tid < N) {
		Particle& particle = particles[tid];
		
		particle.vel += delta_time * force_buffer[tid];
		particle.pos += delta_time * particle.vel;

		// Boundary condition: Dampen and reverse velocity vector
        // Only ceiling
        if (tid >= immovable_particle_num) {
            if (particle.pos.y > max_box_bound.y) {
                particle.pos.y = max_box_bound.y;
                particle.vel.y = -particle.vel.y * damping;
            }
        }

	}

}

/* Leapfrog integration scheme */

/* Performs integration step for position and half step for velocity */
__global__ void
leapfrog_pre_integration(Particle* particles, float3* force_buffer, float mass_inv, float delta_time, int N, int immovable_particle_num, int dam_particle_num, float3 min_box_bound, float3 max_box_bound, float damping) {

    int tid = (blockIdx.x * blockDim.x) + threadIdx.x + immovable_particle_num - dam_particle_num;

    if (tid >= immovable_particle_num - dam_particle_num && tid < N) {
        Particle& particle = particles[tid];

        particle.pos += delta_time * particle.vel + 0.5 * force_buffer[tid] * delta_time * delta_time;
        particle.vel += 0.5 * force_buffer[tid] * delta_time;

        // Boundary condition: Dampen and reverse velocity vector
        // Only ceiling
        if (tid >= immovable_particle_num) {
            if (particle.pos.y > max_box_bound.y) {
                particle.pos.y = max_box_bound.y;
            }
        }

    }
}

/* Performs half integration step for velocity */
__global__ void
leapfrog_post_integration(Particle* particles, float3* force_buffer, float mass_inv, float delta_time, int N, int immovable_particle_num, int dam_particle_num, float3 min_box_bound, float3 max_box_bound, float damping) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x + immovable_particle_num - dam_particle_num;

    if (tid >= immovable_particle_num - dam_particle_num && tid < N) {
        Particle& particle = particles[tid];
        particle.vel += 0.5 * force_buffer[tid] * delta_time;

        if (tid >= immovable_particle_num) {
            if (particle.pos.y == max_box_bound.y) {
                particle.vel.y = -particle.vel.y * damping;
            }
        }
    }

}