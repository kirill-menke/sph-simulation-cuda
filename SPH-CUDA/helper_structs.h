#pragma once
#define _USE_MATH_DEFINES
#include <cmath>
#include <unordered_map>
#include <string>

#include "cuda_runtime.h"

#include "helper_math.h"


enum class Integrator {
	ForwardEuler,
	Leapfrog
};


struct Parameters {

	int movable_particle_num;
	int immovable_particle_num;
	int dam_particle_num;
	float time_step;

	/* Integration method */
	Integrator integrator;

	/* Smoothing radius */
	float h;
	float h_inv;
	float h2;
	float h3;

	/* Kernel constants */
	float const_poly6;
	float const_spiky;
	float const_visc;
	float const_surf;

	/* Pressure constant / stiffness */
	float k;
	/* Reference pressure */
	float p0;
	/* Viscosity constant */
	float e;
	/* Surface tension constant */
	float s;
	/* Particle mass */
	float mass;
	float mass_inv;
	/* Gravity */
	float3 g;

	/* Box parameters */
	float damping;
	float3 min_box_bound;
	float3 max_box_bound;
	float3 cell_dims;
	float3 cell_size;
	int cell_num;
	int3 particle_depth_per_dim; // unit: particles
	float wall_density;

	/* GPU parameters */
	int threads_per_group;
	int thread_groups_part;
	int thread_groups_cell;

	/* Particle spawn parameters */
	float spawn_dist;		// unit: coordinates
	float boundary_spawn_dist;
	int edge_length;		// unit: particles
	float3 spawn_offset;	// unit: coordinates

	/* Visualization parameters */
	float particle_radius;
	int draw_number;

	//#define RENDERWALLS

	Parameters(std::unordered_map<std::string, std::string> params) :
		movable_particle_num(std::stoi(params["movable_particle_num"])),
		time_step(std::stof(params["timestep"])),

		integrator(params["integrator"] == "euler" ? Integrator::ForwardEuler : Integrator::Leapfrog),

		const_poly6(315 / (64 * float(M_PI) * powf(h, 9))),
		const_spiky(-45 / (float(M_PI) * powf(h, 6))),
		const_visc(15 / (2 * float(M_PI) * powf(h, 3))),
		const_surf(3 / (2 * float(M_PI) * powf(h, 3))),

		h(std::stof(params["h"])), h_inv(1 / h), h2(h* h), h3(h* h* h),
		k(std::stof(params["k"])), p0(std::stof(params["p0"])), e(std::stof(params["e"])), s(std::stof(params["s"])), 
		mass(std::stof(params["mass"])), g(make_float3(0, std::stof(params["g"]), 0)), mass_inv(1. / mass),

		damping(std::stof(params["boundary_damping"])),
		wall_density(std::stof(params["wall_density"])),
		min_box_bound(make_float3(std::stof(params["min_box_x"]), std::stof(params["min_box_y"]), std::stof(params["min_box_z"]))),
		max_box_bound(make_float3(std::stof(params["max_box_x"]), std::stof(params["max_box_y"]), std::stof(params["max_box_z"]))),
		cell_dims(((max_box_bound - min_box_bound) / h) + 1.),
		cell_num(int(cell_dims.x* cell_dims.y* cell_dims.z)),

		threads_per_group(std::stoi(params["threads_per_group"])),
		thread_groups_cell(int((cell_num + threads_per_group - 1) / threads_per_group)),

		spawn_dist(std::stof(params["spawn_dist"])),
		boundary_spawn_dist(std::stof(params["boundary_spawn_dist"])),
		edge_length(powf(movable_particle_num, 1./3.)),
		spawn_offset(make_float3(std::stof(params["spawn_off_x"]), std::stof(params["spawn_off_y"]), std::stof(params["spawn_off_z"]))),

		particle_radius(std::stof(params["particle_radius"]))
	{
		/* Calculate number of particles per wall */
		float3 box_dim = max_box_bound - min_box_bound;
		particle_depth_per_dim = ceil(box_dim / boundary_spawn_dist);
		// Calculate number of particles for the five boundary walls (without ceiling)
		immovable_particle_num = particle_depth_per_dim.x * particle_depth_per_dim.y * 2 + particle_depth_per_dim.z * particle_depth_per_dim.y * 2 + particle_depth_per_dim.z * particle_depth_per_dim.x;
		// Calculate number of particles for dam
		dam_particle_num = particle_depth_per_dim.x * particle_depth_per_dim.y;
		// Add to update total number of immovable particles
		immovable_particle_num += dam_particle_num;

		// Choose number of threads as maximum of immovable and moveable particles
		int thread_count = max(immovable_particle_num, movable_particle_num);
		thread_groups_part = int((thread_count + threads_per_group - 1) / threads_per_group);

		#if defined(RENDERWALLS)
			draw_number = immovable_particle_num + movable_particle_num;
		#else
			draw_number = movable_particle_num + dam_particle_num;
		#endif
	}
};


struct Particle {
	float3 pos;
	float3 vel;

	Particle(float3 pos, float3 vel) : 
		pos(pos), vel(vel) {}
};