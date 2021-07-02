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

	int particle_num;
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

	/* GPU parameters */
	int threads_per_group;
	int thread_groups_part;
	int thread_groups_cell;

	/* Particle spawn parameters */
	float spawn_dist;		// unit: coordinates
	int edge_length;		// unit: particles
	float3 spawn_offset;	// unit: coordinates

	/* Visualization parameters */
	float particle_radius;


	Parameters(std::unordered_map<std::string, std::string> params) :
		particle_num(std::stoi(params["particle_num"])),
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
		min_box_bound(make_float3(std::stof(params["min_box_x"]), std::stof(params["min_box_y"]), std::stof(params["min_box_z"]))),
		max_box_bound(make_float3(std::stof(params["max_box_x"]), std::stof(params["max_box_y"]), std::stof(params["max_box_z"]))),
		cell_dims(((max_box_bound - min_box_bound) / h) + 1.),
		cell_size((max_box_bound - min_box_bound) / cell_dims),
		cell_num(int(cell_dims.x* cell_dims.y* cell_dims.z)),

		threads_per_group(std::stoi(params["threads_per_group"])),
		thread_groups_part(int((particle_num + threads_per_group - 1) / threads_per_group)),
		thread_groups_cell(int((cell_num + threads_per_group - 1) / threads_per_group)),

		spawn_dist(std::stof(params["spawn_dist"])),
		edge_length(powf(particle_num, 1./3.)),
		spawn_offset(make_float3(std::stof(params["spawn_off_x"]), std::stof(params["spawn_off_y"]), std::stof(params["spawn_off_z"]))),

		particle_radius(std::stof(params["particle_radius"]))
	{}
};


struct Particle {
	float3 pos;
	float3 vel;

	Particle(float3 pos, float3 vel) : 
		pos(pos), vel(vel) {}
};