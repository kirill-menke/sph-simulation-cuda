# Real-Time SPH Simulation in C++/CUDA

## Description
This project implements a fluid simulation using the particle-based SPH solver for the Navier-Stokes equations.
The simulation (calculation of densities and forces) was written in CUDA and the rendering was done with OpenGL.
We implemented a typical dambreak scenario which supports two rendering modes: Rendering of individual particles and surface rendering based on Marching Cubes. 

The video below shows a real-time simulation of about 50.000 particles on the GeForce GTX 1080 Ti achieving stable 70 FPS in particle rendering mode and 40 FPS in surface rendering mode.

<img src="recordings/dambreak.gif" alt="dambreak_scenario" width="1000"/>


## Implementation
For SPH the simulation domain is discretized with particles which carry certain properties. In our case, we store the position, velocity, and density for every particle.
From these quantities, the forces acting on the individual particles can be calculated in every timestep to advect them forward in time.
In this implementation only the pressure force, viscosity force, and gravity force were considered. 
To calculate these forces acting on each particle, SPH considers all surrounding particles within a so-called smoothing radius.
Here, a naive algorithm that checks every other particle if it lies within this radius would have a runtime of O(n²).
Hence, we utilized an uniform grid for the neighborhood search which divides the simulation domain into equally-sized cells. In this way, only neighboring cells need to be checked for potential neighbors, reducing the runtime to O(n).
### Simulation Pipeline
The following pipeline summarizes the step which were perfomed in every timestep to solve the 
![simulation_pipeline](visualization/simulation_pipeline.png)

1. ***Insert Particles***

    Particles are inserted into the uniform grid, i.e. each particle is assigned to one cell. Each cell has the size of the smoothing radius used for SPH.

2. ***Compute Pressures***
   
    Based on the grid, the density at every particles location is calculated. To do this, we simply iterate over each particle and look in its neighbouring cells for other particles that lie within the smoothing radius. Those who do, are included into the density calculation, i.e. only particles within this smoothin radius have influence on the current density. Using the density, we can then calculate the pressure required for the pressure force.

    ![density_formula](visualization/density.png) ![pressure_formula](visualization/pressure_calculation.png)

1. 

### Boundary Handling

## Code Structure
Starting the application
Parameters files
You can switch between these modes using the `space` key.


# Prerequisites and build
CUDA 11.6
Visual Studio
dependencies related to OpenGL
Rendering related code can be found in ... 