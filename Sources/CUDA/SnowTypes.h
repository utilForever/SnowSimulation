/*************************************************************************
> File Name: SnowTypes.h
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Snow type functions compatibles with CUDA.
> Created Time: 2018/01/14
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#ifndef SNOW_SIMULATION_SNOW_TYPES_H
#define SNOW_SIMULATION_SNOW_TYPES_H

#define CUDA_INCLUDE
#include <Common/Math.h>
#include <CUDA/Noise.h>
#include <CUDA/Vector.h>
#include <Simulation/Particle.h>
#include <Simulation/Material.h>

#include <cuda.h>
#include <math.h>
#include <device_launch_parameters.h>

/*
* theta_c, theta_s -> determine when snow starts breaking.
*          larger = chunky, wet. smaller = powdery, dry
*
* low xi, E0 = muddy. high xi, E0 = Icy
* low xi = ductile, high xi = brittle
*
*/
__global__
void ApplyChunky(Particle* particles, int particleCount)
{
	// spatially varying constitutive parameters
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= particleCount)
	{
		return;
	}

	Particle& particle = particles[tid];
	Vector3 pos = particle.position;
	// adjust the .5 to get desired frequency of chunks within fbm
	float fbm = FBM3(pos * 30.f);

	Material mat;
	mat.SetYoungsAndPoissons(MIN_E0 + fbm * (MAX_E0 - MIN_E0), POISSONS_RATIO);
	mat.xi = MIN_XI + fbm * (MAX_XI - MIN_XI);
	mat.SetCriticalStrains(5e-4, 1e-4);

	particle.material = mat;
}

#endif