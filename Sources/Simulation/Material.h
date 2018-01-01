/*************************************************************************
> File Name: Material.h
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Material structure of snow simulation.
> Created Time: 2018/01/01
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#ifndef SNOW_SIMULATION_MATERIAL_H
#define SNOW_SIMULATION_MATERIAL_H

#include <cuda.h>
#include <cuda_runtime.h>

constexpr float POISSONS_RATIO = 0.2f;

// Youngs modulus
constexpr float E0 = 1.4e5f;
constexpr float MIN_E0 = 4.8e4f;
constexpr float MAX_E0 = 1.4e5f;

// Critical compression
constexpr float MIN_THETA_C = 1.9e-2f;
constexpr float MAX_THETA_C = 2.5e-2f;

// Critical stretch
constexpr float MIN_THETA_S = 5e-3f;
constexpr float MAX_THETA_S = 7.5e-3f;

// Hardening coefficient
constexpr int MIN_XI = 5;
constexpr int MAX_XI = 10;

struct Material
{
	// first Lame parameter
	float lambda; 
	// second Lame parameter
	float mu; 
	// Plastic hardening parameter
	float xi;

	// Singular values restricted to [criticalCompression, criticalStretch]
	float criticalCompressionRatio;
	float criticalStretchRatio;

	// Constants from paper
	__host__ __device__
	Material()
	{
		SetYoungsAndPoissons(E0, POISSONS_RATIO);
		xi = 10;
		SetCriticalStrains(MAX_THETA_C, MAX_THETA_S);
	}

	__host__ __device__
	Material(float compression, float stretch, float hardeningCoeff, float youngsModulus) :
		xi(hardeningCoeff), criticalCompressionRatio(compression), criticalStretchRatio(stretch)
	{
		SetYoungsAndPoissons(youngsModulus, POISSONS_RATIO);
	}

	// Set constants in terms of Young's modulus and Poisson's ratio
	__host__ __device__
	void SetYoungsAndPoissons(float E, float v)
	{
		lambda = (E * v) / ((1 + v) * (1 - 2 * v));
		mu = E / (2 * (1 + v));
	}

	// Set constants in terms of Young's modulus and shear modulus (mu)
	__host__ __device__
	void SetYoungsAndShear(float E, float G)
	{
		lambda = G * (E - 2 * G) / (3 * G - E);
		mu = G;
	}

	// Set constants in terms of Lame's first parameter (lambda) and shear modulus (mu)
	__host__ __device__
	void SetLameAndShear(float L, float G)
	{
		lambda = L;
		mu = G;
	}

	// Set constants in terms of Lame's first parameter (lambda) and Poisson's ratio
	__host__ __device__
	void SetLameAndPoissons(float L, float v)
	{
		lambda = L;
		mu = L * (1 - 2 * v) / (2 * v);
	}

	// Set constants in terms of shear modulus (mu) and Poisson's ratio
	__host__ __device__
	void SetShearAndPoissons(float G, float v)
	{
		lambda = (2 * G * v) / (1 - 2 * v);
		mu = G;
	}

	__host__ __device__
	void SetCriticalCompressionStrain(float thetaC)
	{
		criticalCompressionRatio = 1.f - thetaC;
	}

	__host__ __device__
	void SetCriticalStretchStrain(float thetaS)
	{
		criticalStretchRatio = 1.f + thetaS;
	}

	__host__ __device__
	void SetCriticalStrains(float thetaC, float thetaS)
	{
		criticalCompressionRatio = 1.f - thetaC;
		criticalStretchRatio = 1.f + thetaS;
	}
};

#endif