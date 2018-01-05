/*************************************************************************
> File Name: Functions.h
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Functions compatibles with CUDA.
> Created Time: 2018/01/06
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#ifndef SNOW_SIMULATION_FUNCTIONS_H
#define SNOW_SIMULATION_FUNCTIONS_H

#include <Simulation/Node.h>

using GLuint = unsigned int;

struct cudaGraphicsResource;
struct Grid;
struct Particle;
struct ParticleCache;
struct Node;
struct NodeCache;
struct ImplicitCollider;
struct SimulationParameters;
struct Material;

extern "C"
{
	// OpenGL-CUDA interop
	void RegisterVBO(cudaGraphicsResource** resource, GLuint vbo);
	void UnregisterVBO(cudaGraphicsResource* resource);

	// Particle simulation
	void UpdateParticles(
		Particle* particles, ParticleCache* devParticleCache, ParticleCache* hostParticleCache, int numParticles,
		Grid* grid, Node* nodes, NodeCache* nodeCache, int numNodes,
		ImplicitCollider* colliders, int numColliders, float timeStep, bool implicitUpdate);

	// Mesh filling
	void FillMesh(cudaGraphicsResource** resource, int triCount, const Grid& grid, Particle* particles, int particleCount, float targetDensity, int materialPreset);
	
	// One time computation to get particle volumes
	void InitializeParticleVolumes(Particle* particles, int numParticles, const Grid* grid, int numNodes);
}

#endif