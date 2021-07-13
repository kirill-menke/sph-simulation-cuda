#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "helper_structs.h"
#include "file_manager.h"
#include "error_checker.h"

#include "cell_structure.cuh"
#include "calculate_density.cuh"
#include "calculate_force.cuh"
#include "integrators.cuh"

#include "visualizer.h"
#include "particlePositionCopy.cu"
#include <cuda_gl_interop.h>

void initializeWallParticles(std::vector<Particle>&, Parameters&);
void initializeParticles(std::vector<Particle>&, Parameters&);
void placeWallParticles(std::vector<Particle>& particles, Parameters& p, int limit, float3 offset, int3 order);
void initializeDensity(std::vector<float3>& density, Parameters& p);

int main() {
	FileManager file_manager("parameter_files", "params_0.par");
	Parameters params = file_manager.readParams();

	std::vector<Particle> particles;
	initializeWallParticles(particles, params);
	initializeParticles(particles, params);

	std::vector<float3> densities;
	initializeDensity(densities, params);

	// Add particle_num (movable) and dam_particle_num (solid), both types of particles are stored in particles
	size_t total_particle_num = params.movable_particle_num + params.immovable_particle_num;
	std::vector<int> cell_list(params.cell_num, -1);
	std::vector<int> particle_list(total_particle_num, -1);

	/* Allocate memory on device */
	Particle* d_particles;
	float3* d_force_buffer;
	float* d_density_buffer;
	int* d_particle_list, * d_cell_list;
	size_t bytes_vec = sizeof(float) * total_particle_num;
	size_t bytes_vec3 = sizeof(float3) * total_particle_num;
	size_t bytes_struct = sizeof(Particle) * total_particle_num;
	size_t bytes_particle_list = sizeof(int) * total_particle_num;
	size_t bytes_cell_list = sizeof(int) * params.cell_num;

	checkError(cudaMalloc((void**)&d_particle_list, bytes_particle_list));
	checkError(cudaMalloc((void**)&d_cell_list, bytes_cell_list));
	checkError(cudaMalloc((void**)&d_particles, bytes_struct));
	checkError(cudaMalloc(&d_force_buffer, bytes_vec3));
	checkError(cudaMalloc((void**)&d_density_buffer, bytes_vec));

	/* Copy data to device */
	checkError(cudaMemcpy(d_particles, particles.data(), bytes_struct, cudaMemcpyHostToDevice));
	checkError(cudaMemcpy(d_particle_list, particle_list.data(), bytes_particle_list, cudaMemcpyHostToDevice));
	checkError(cudaMemcpy(d_cell_list, cell_list.data(), bytes_cell_list, cudaMemcpyHostToDevice));
	checkError(cudaMemcpy(d_density_buffer, densities.data(), bytes_vec, cudaMemcpyHostToDevice));

	/* Visualization init */
	Visualizer vis(params.draw_number, params.particle_radius, params.min_box_bound.x, params.min_box_bound.y, params.min_box_bound.z,
		params.max_box_bound.x, params.max_box_bound.y, params.max_box_bound.z);

	struct cudaGraphicsResource* positionsVBO_CUDA = NULL;
	checkError(cudaGraphicsGLRegisterBuffer(&positionsVBO_CUDA, vis.vertexArray, cudaGraphicsMapFlagsWriteDiscard));
	
	if (params.integrator == Integrator::Leapfrog) {
		/* Initialize cell list and particle list */
		assign_to_cells << <params.thread_groups_part, params.threads_per_group >> > (d_particles, d_cell_list, d_particle_list,
			total_particle_num, params.immovable_particle_num, params.cell_dims, params.min_box_bound, params.h_inv);
		checkError(cudaPeekAtLastError());
		checkError(cudaDeviceSynchronize());

		/* Calculate densities */
		calculate_density << <params.thread_groups_part, params.threads_per_group >> > (d_particles, d_cell_list, d_particle_list, d_density_buffer,
			params.cell_dims, params.min_box_bound, total_particle_num, params.immovable_particle_num, params.h, params.h2, params.h_inv, params.const_poly6, params.mass, params.p0);
		checkError(cudaPeekAtLastError());
		checkError(cudaDeviceSynchronize());

		/* Calculate forces */
		calculate_force << <params.thread_groups_part, params.threads_per_group >> > (d_particles, d_cell_list, d_particle_list, d_force_buffer, d_density_buffer, params.cell_dims, params.min_box_bound,
			total_particle_num, params.immovable_particle_num, params.h, params.h_inv, params.const_spiky, params.const_visc, params.const_surf, params.mass, params.k, params.e, params.p0, params.s, params.g);
		checkError(cudaPeekAtLastError());
		checkError(cudaDeviceSynchronize());
	}
	
	std::cout << "Simulation started" << std::endl;
	std::cout << params.spawn_dist << std::endl;
	while (!glfwWindowShouldClose(vis.window)) {
		if (vis.runSimulation) {
			// Start time measurement
			std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

			if (params.integrator == Integrator::Leapfrog) {
				/* Integrate position and velocity */
				leapfrog_pre_integration << <params.thread_groups_part, params.threads_per_group >> > (d_particles, d_force_buffer, params.mass_inv, params.time_step,
					total_particle_num, params.immovable_particle_num, params.min_box_bound, params.max_box_bound, params.damping);
			}

			/* Set all entries of cell list to -1 */
			reset_cell_list << <params.thread_groups_cell, params.threads_per_group >> > (d_cell_list, params.cell_num);
			checkError(cudaPeekAtLastError());
			checkError(cudaDeviceSynchronize());

			/* Initialize cell list and particle list */
			assign_to_cells << <params.thread_groups_part, params.threads_per_group >> > (d_particles, d_cell_list, d_particle_list,
				total_particle_num, params.immovable_particle_num, params.cell_dims, params.min_box_bound, params.h_inv);
			checkError(cudaPeekAtLastError());
			checkError(cudaDeviceSynchronize());

			/* Calculate densities */
			calculate_density << <params.thread_groups_part, params.threads_per_group >> > (d_particles, d_cell_list, d_particle_list, d_density_buffer,
				params.cell_dims, params.min_box_bound, total_particle_num, params.immovable_particle_num, params.h, params.h2, params.h_inv, params.const_poly6, params.mass, params.p0);
			checkError(cudaPeekAtLastError());
			checkError(cudaDeviceSynchronize());

			/* Calculate forces */
			calculate_force << <params.thread_groups_part, params.threads_per_group >> > (d_particles, d_cell_list, d_particle_list, d_force_buffer, d_density_buffer, params.cell_dims, params.min_box_bound,
				total_particle_num, params.immovable_particle_num, params.h, params.h_inv, params.const_spiky, params.const_visc, params.const_surf, params.mass, params.k, params.e, params.p0, params.s, params.g);
			checkError(cudaPeekAtLastError());
			checkError(cudaDeviceSynchronize());

			if (params.integrator == Integrator::Leapfrog) {
				/* Integrate new positions and velocities */
				leapfrog_post_integration << <params.thread_groups_part, params.threads_per_group >> >
					(d_particles, d_force_buffer, params.mass_inv, params.time_step, total_particle_num, params.immovable_particle_num, params.min_box_bound, params.max_box_bound, params.damping);
			}
			else {
				/* Integrate new positions and velocities */
				integrate_symplectic_euler << <params.thread_groups_part, params.threads_per_group >> >
					(d_particles, d_force_buffer, params.time_step, total_particle_num, params.immovable_particle_num, params.min_box_bound, params.max_box_bound, params.damping);
				checkError(cudaPeekAtLastError());
				checkError(cudaDeviceSynchronize());
			}

			// Stop time measurement
			std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
			std::cout << "Time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;

			/* Visualization update */
			float* vertexPointer;
			// Map the buffer to CUDA
			checkError(cudaGraphicsMapResources(1, &positionsVBO_CUDA));
			size_t numBytes;
			checkError(cudaGraphicsResourceGetMappedPointer((void**)&vertexPointer, &numBytes, positionsVBO_CUDA));
			// Run kernel
			copy_particle_positions << <params.thread_groups_part, params.threads_per_group >> > ((float*)vertexPointer, d_particles, total_particle_num, params.immovable_particle_num);
			// Unmap the buffer
			checkError(cudaGraphicsUnmapResources(1, &positionsVBO_CUDA));

			vis.draw(params.draw_number);

			// Stop time measurement
			end = std::chrono::steady_clock::now();
			std::cout << "Time2 = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;

		}
	}
	std::cout << "Simulation finished" << std::endl;

	/* Free memory on device */
	checkError(cudaFree(d_particles));
	checkError(cudaFree(d_force_buffer));
	checkError(cudaFree(d_density_buffer));
	checkError(cudaFree(d_particle_list));
	checkError(cudaFree(d_cell_list));

	/* Visualization end */
	vis.end();
}

/* Determines position of wall particles, should be called before initilizing moveable particles */
void initializeWallParticles(std::vector<Particle>& particles, Parameters& p) {
	// Calculate shift in order to spawn the cubic shape in the center of the box
	// Shift equals half of the length of the cubic shape
	float shift = p.particle_radius;
	int wall_num = 5;
	int limit;
	float3 offset;

	// boundary bottom
	offset = make_float3(p.min_box_bound.x + shift, p.min_box_bound.z + shift, p.min_box_bound.y + shift);
	limit = p.particle_depth_per_dim.z * p.particle_depth_per_dim.x;
	placeWallParticles(particles, p, limit, offset, make_int3(0, 2, 1));

	// boundary in "false" z front
	offset = make_float3(p.min_box_bound.z + shift, p.min_box_bound.y + shift, p.min_box_bound.x + shift);
	limit = p.particle_depth_per_dim.z* p.particle_depth_per_dim.y;
	placeWallParticles(particles, p, limit, offset, make_int3(2, 1, 0));

	// boundary in "false" z back
	offset = make_float3(p.min_box_bound.z + shift, p.min_box_bound.y + shift, p.max_box_bound.x - shift);
	limit = p.particle_depth_per_dim.z * p.particle_depth_per_dim.y;
	placeWallParticles(particles, p, limit, offset, make_int3(2, 1, 0));

	// boundary in "false" x right
	offset = make_float3(p.min_box_bound.x + shift, p.min_box_bound.y + shift, p.max_box_bound.z - shift);
	limit = p.particle_depth_per_dim.x * p.particle_depth_per_dim.y;
	placeWallParticles(particles, p, limit, offset, make_int3(0, 1, 2));

	// boundary in "false" x left
	offset = make_float3(p.min_box_bound.x + shift, p.min_box_bound.y + shift, p.min_box_bound.z + shift);
	limit = p.particle_depth_per_dim.x * p.particle_depth_per_dim.y;
	placeWallParticles(particles, p, limit, offset, make_int3(0, 1, 2));

}

void placeWallParticles(std::vector<Particle>& particles, Parameters& p, int limit, float3 offset, int3 order) {
	float coordiantes[3];
	int particle_depths[3] = {p.particle_depth_per_dim.x, p.particle_depth_per_dim.y, p.particle_depth_per_dim.z};

	for (int i = 0; i < limit; i++) {

		// Calculate wall shape
		coordiantes[0] = (i % particle_depths[order.x]) * p.boundary_spawn_dist;
		coordiantes[1] = floor((i / particle_depths[order.x])) * p.boundary_spawn_dist;

		// Add offset
		coordiantes[0] += offset.x;
		coordiantes[1] += offset.y;
		coordiantes[2] = offset.z;

		float3 position = make_float3(coordiantes[order.x], coordiantes[order.y], coordiantes[order.z]);
		particles.emplace_back(position, make_float3(0., 0., 0.));
	}
}

/* Spawns particles in a cubic shape */
void initializeParticles(std::vector<Particle>& particles, Parameters& p) {
	// Calculate shift in order to spawn the cubic shape in the center of the box
	// Shift equals half of the length of the cubic shape
	float shift = (p.edge_length * p.spawn_dist) / 2;

	for (int i = 0; i < p.movable_particle_num; i++) {
		
		// Calculate cubic shape
		float x = (i % p.edge_length) * p.spawn_dist;
		float y = ((i / p.edge_length) % p.edge_length) * p.spawn_dist;
		float z = (i / (p.edge_length * p.edge_length)) * p.spawn_dist;

		// Add offsets
		x += p.spawn_offset.x - shift;
		y += p.spawn_offset.y - shift;
		z += p.spawn_offset.z - shift;

		particles.emplace_back(make_float3(x, y, z), make_float3(0., 0., 0.));
	}
}

void initializeDensity(std::vector<float3>& density, Parameters& p) {

	for (int i = 0; i < p.immovable_particle_num; i++) {
		density.emplace_back(make_float3(p.wall_density, p.wall_density, p.wall_density));
	}
}