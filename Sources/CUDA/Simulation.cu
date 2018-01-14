/*************************************************************************
> File Name: Simulation.cu
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Simulation CUDA functions.
> Created Time: 2018/01/13
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#ifndef SNOW_SIMULATION_SIMULATION_CU
#define SNOW_SIMULATION_SIMULATION_CU

#include <Common/Math.h>
#include <CUDA/Atomic.h>
#include <CUDA/Collider.h>
#include <CUDA/Decomposition.h>
#include <CUDA/Functions.h>
#include <CUDA/Helpers.h>
#include <CUDA/Implicit.h>
#include <CUDA/Weighting.h>
#include <Simulation/Caches.h>
#include <Simulation/ImplicitCollider.h>
#include <Simulation/Material.h>
#include <Simulation/Node.h>
#include <Simulation/Particle.h>

#include <cuda.h>
#include <math.h>

constexpr float ALPHA = 0.05f;
#define GRAVITY Vector3(0.f, -9.8f, 0.f)

// Chain to compute the volume of the particle
// Part of one time operation to compute particle volumes. First rasterize particle masses to grid
// Operation done over Particles over grid node particle affects
__global__
void ComputeNodeMasses(const Particle* particles, int numParticles, const Grid* grid, float* nodeMasses)
{
	int particleIdx = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
	if (particleIdx >= numParticles)
	{
		return;
	}

	const Particle& particle = particles[particleIdx];

	glm::ivec3 currIJK;
	Grid::GridIndexToIJK( threadIdx.y, glm::ivec3(4,4,4), currIJK );
	Vector3 particleGridPos = (particle.position - grid->pos) / grid->h;
	currIJK += glm::ivec3(particleGridPos - 1);

	if (Grid::WithinBoundsInclusive(currIJK, glm::ivec3(0, 0, 0), grid->dim))
	{
		Vector3 nodePosition(currIJK);
		Vector3 dx = Vector3::Abs(particleGridPos - nodePosition);
		float w = Weight(dx);
		
		atomicAdd(&nodeMasses[Grid::GetGridIndex(currIJK, grid->dim + 1)], particle.mass * w);
	 }
}

// Computes the particle's density * grid's volume. This needs to be separate from computeCellMasses(...) because
// we need to wait for ALL threads to sync before computing the density
// Operation done over Particles over grid node particle affects
 __global__
void ComputeParticleDensity(Particle* particles, int numParticles, const Grid* grid, const float* cellMasses)
{
	int particleIdx = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
	if (particleIdx >= numParticles)
	{
		return;
	}

	Particle& particle = particles[particleIdx];

	glm::ivec3 currIJK;
	Grid::GridIndexToIJK(threadIdx.y, glm::ivec3(4,4,4), currIJK);
	Vector3 particleGridPos = (particle.position - grid->pos) / grid->h;
	currIJK += glm::ivec3(particleGridPos - 1);

	if (Grid::WithinBoundsInclusive(currIJK, glm::ivec3(0,0,0), grid->dim))
	{
		Vector3 nodePosition(currIJK);
		Vector3 dx = Vector3::Abs(particleGridPos - nodePosition);
		float w = Weight(dx);
		float gridVolume = grid->h * grid->h * grid->h;

		// fill volume with particle density. Then in final step, compute volume
        atomicAdd(&particle.volume, cellMasses[Grid::GetGridIndex(currIJK, grid->dim+1)] * w / gridVolume);
	 }
}
 
// Computes the particle's volume. Assumes computeParticleDensity(...) has just been called.
// Operation done over particles
__global__
void ComputeParticleVolume(Particle* particleData, int numParticles)
{
	int particleIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleIdx >= numParticles)
	{
		return;
	}

	// Note: particle.volume is assumed to be the (particle's density ) before we compute it correctly
	Particle& particle = particleData[particleIdx];
	particle.volume = particle.mass / particle.volume;
}

__host__
void InitializeParticleVolumes(Particle* particles, int numParticles, const Grid* grid, int numNodes)
{
	float* devNodeMasses;
	cudaMalloc(reinterpret_cast<void**>(&devNodeMasses), numNodes * sizeof(float));
	cudaMemset(devNodeMasses, 0, numNodes * sizeof(float));

	const dim3 blocks((numParticles + THREAD_COUNT - 1) / THREAD_COUNT, 64);
	static const dim3 threads(THREAD_COUNT / 64, 64);

	LAUNCH(ComputeNodeMasses<<<blocks, threads>>>(particles, numParticles, grid, devNodeMasses));

	LAUNCH(ComputeParticleDensity<<<blocks, threads>>>(particles, numParticles, grid, devNodeMasses));

	LAUNCH(ComputeParticleVolume<<<(numParticles + THREAD_COUNT - 1) / THREAD_COUNT, THREAD_COUNT>>>(particles, numParticles));

	cudaFree(devNodeMasses);
}

__global__
void ComputeSigma(const Particle* particles, ParticleCache* particleCache, int numParticles, const Grid* grid)
{
	int particleIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleIdx >= numParticles)
	{
		return;
	}

	const Particle& particle = particles[particleIdx];

	// for the sake of making the code look like the math
	const Matrix3& Fp = particle.plasticF;
	const Matrix3& Fe = particle.elasticF;

	float Jpp = Matrix3::Determinant(Fp);
	float Jep = Matrix3::Determinant(Fe);

	Matrix3 Re;
	ComputePD(Fe, Re);

	const Material material = particle.material;

	float muFp = material.mu * expf(material.xi * (1 - Jpp));
	float lambdaFp = material.lambda * expf(material.xi * (1 - Jpp));

	particleCache->sigmas[particleIdx] = (2 * muFp * Matrix3::MultiplyABT(Fe - Re, Fe) + Matrix3(lambdaFp * (Jep - 1) * Jep)) * -particle.volume;
}

/**
 * Called on each particle.
 *
 * Each particle adds it's mass, velocity and force contribution to the grid nodes within 2h of itself.
 *
 * In:
 * particleData -- list of particles
 * grid -- Stores grid paramters
 * worldParams -- Global parameters dealing with the physics of the world
 *
 * Out:
 * nodes -- list of every node in grid ((dim.x+1)*(dim.y+1)*(dim.z+1))
 *
 */
__global__
void ComputeCellMassVelocityAndForceFast(const Particle* particleData, const ParticleCache* particleCache, int numParticles, const Grid* grid, Node* nodes)
{
	int particleIdx = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
	if (particleIdx >= numParticles)
	{
		return;
	}

	const Particle& particle = particleData[particleIdx];

	glm::ivec3 currIJK;
	Grid::GridIndexToIJK(threadIdx.y, glm::ivec3(4,4,4), currIJK);
	Vector3 particleGridPos = (particle.position - grid->pos) / grid->h;
	currIJK += glm::ivec3(particleGridPos - 1);

	if (Grid::WithinBoundsInclusive(currIJK, glm::ivec3(0, 0, 0), grid->dim))
	{
		Node& node = nodes[Grid::GetGridIndex(currIJK, grid->dim + 1)];

		float w;
		Vector3 wg;
		Vector3 nodePosition(currIJK.x, currIJK.y, currIJK.z);
		WeightAndGradient(particleGridPos - nodePosition, w, wg);

        atomicAdd(&node.mass, particle.mass * w);
        atomicAdd(&node.velocity, particle.velocity * particle.mass * w);
        atomicAdd(&node.force, particleCache->sigmas[particleIdx] * wg);
	}
}

/**
 * Called on each grid node.
 *
 * Updates the velocities of each grid node based on forces and collisions
 *
 * In:
 * nodes -- list of all nodes in the grid.
 * dt -- delta time, time step of simulation
 * colliders -- array of colliders in the scene.
 * numColliders -- number of colliders in the scene
 * worldParams -- Global parameters dealing with the physics of the world
 * grid -- parameters defining the grid
 *
 * Out:
 * nodes -- updated velocity and velocityChange
 *
 */
__global__
void UpdateNodeVelocities(Node* nodes, int numNodes, float dt, const ImplicitCollider* colliders, int numColliders, const Grid* grid, bool updateVelocityChange)
{
	int nodeIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (nodeIdx >= numNodes)
	{
		return;
	}

	Node& node = nodes[nodeIdx];

	if (node.mass > 0.f)
	{
		// Have to normalize velocity by mass to conserve momentum
		float scale = 1.f / node.mass;
		node.velocity *= scale;

		// Initialize velocityChange with pre-update velocity
		node.velocityChange = node.velocity;

		// Gravity for node forces
		node.force += node.mass * GRAVITY;

		// Update velocity with node force
		node.velocity += dt * scale * node.force;

		// Handle collisions
		int gridI, gridJ, gridK;
		Grid::GridIndexToIJK(nodeIdx, gridI, gridJ, gridK, grid->dim + 1);
		Vector3 nodePosition = Vector3(gridI, gridJ, gridK) * grid->h + grid->pos;
		CheckForAndHandleCollisions(colliders, numColliders, nodePosition, node.velocity);

		if (updateVelocityChange)
		{
			node.velocityChange = node.velocity - node.velocityChange;
		}
	}
}

// Use weighting functions to compute particle velocity gradient and update particle velocity
__device__
void ProcessGridVelocities(Particle& particle, const Grid* grid, const Node* nodes, Matrix3& velocityGradient)
{
	const Vector3& pos = particle.position;
	const glm::ivec3& dim = grid->dim;
	const float h = grid->h;

	// Compute neighborhood of particle in grid
	Vector3 particleGridPos = (pos - grid->pos) / h;
	Vector3 gridMax = Vector3::Floor(particleGridPos + Vector3(2, 2, 2));
	Vector3 gridMin = Vector3::Ceil(particleGridPos - Vector3(2, 2, 2));
	glm::ivec3 maxIndex = glm::clamp(glm::ivec3(gridMax), glm::ivec3(0, 0, 0), dim);
	glm::ivec3 minIndex = glm::clamp(glm::ivec3(gridMin), glm::ivec3(0, 0, 0), dim);

	// For computing particle velocity gradient:
	//      grad(v_p) = sum( v_i * transpose(grad(w_ip)) ) = [3x3 matrix]
	// For updating particle velocity:
	//      v_PIC = sum( v_i * w_ip )
	//      v_FLIP = v_p + sum( dv_i * w_ip )
	//      v = (1-alpha)*v_PIC _ alpha*v_FLIP
	Vector3 vPIC(0,0,0), dvFLIP(0,0,0);
	int rowSize = dim.z + 1;
	int pageSize = (dim.y + 1) * rowSize;

	for (int i = minIndex.x; i <= maxIndex.x; ++i)
	{
		Vector3 d, s;
		d.x = particleGridPos.x - i;
		d.x *= (s.x = (d.x < 0) ? -1.f : 1.f);
		
		int pageOffset = i * pageSize;

		for (int j = minIndex.y; j <= maxIndex.y; ++j)
		{
			d.y = particleGridPos.y - j;
			d.y *= (s.y = (d.y < 0) ? -1.f : 1.f);

			int rowOffset = pageOffset + j * rowSize;
			
			for (int k = minIndex.z; k <= maxIndex.z; ++k)
			{
				d.z = particleGridPos.z - k;
				d.z *= (s.z = (d.z < 0) ? -1.f : 1.f);
				
				const Node& node = nodes[rowOffset + k];
				float w;
				Vector3 wg;
				
				WeightAndGradient(s, d, w, wg);
				velocityGradient += Matrix3::OuterProduct(node.velocity, wg);
				
				// Particle velocities
				vPIC += node.velocity * w;
				dvFLIP += node.velocityChange * w;
			}
		}
	}

	particle.velocity = (1.f - ALPHA) * vPIC + ALPHA * (particle.velocity + dvFLIP);
}

__device__
void UpdateParticleDeformationGradients(Particle& particle, const Matrix3& velocityGradient, float timeStep)
{
	// Temporarily assign all deformation to elastic portion
	particle.elasticF = Matrix3::AddIdentity(timeStep * velocityGradient) * particle.elasticF;
	const Material& material = particle.material;

	// Clamp the singular values
	Matrix3 W, S, Sinv, V;
	ComputeSVD(particle.elasticF, W, S, V);

	// FAST COMPUTATION:
	S = Matrix3(
		Clamp(S[0], material.criticalCompressionRatio, material.criticalStretchRatio), 0.f, 0.f,
		0.f, Clamp(S[4], material.criticalCompressionRatio, material.criticalStretchRatio), 0.f,
		0.f, 0.f, Clamp(S[8], material.criticalCompressionRatio, material.criticalStretchRatio));
	Sinv = Matrix3(
		1.f / S[0], 0.f, 0.f,
		0.f, 1.f / S[4], 0.f,
		0.f, 0.f, 1.f / S[8]);
	particle.plasticF = Matrix3::MultiplyADBT(V, Sinv, W) * particle.elasticF * particle.plasticF;
	particle.elasticF = Matrix3::MultiplyADBT(W, S, V);
}

__global__
void UpdateParticlesFromGrid(Particle* particles, int numParticles, const Grid* grid, const Node* nodes, float timeStep, const ImplicitCollider* colliders, int numColliders)
{
	int particleIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (particleIdx >= numParticles)
	{
		return;
	}

	Particle& particle = particles[particleIdx];

	// Update particle velocities and fill in velocity gradient for deformation gradient computation
	Matrix3 velocityGradient = Matrix3(0.f);
	ProcessGridVelocities(particle, grid, nodes, velocityGradient);

	UpdateParticleDeformationGradients(particle, velocityGradient, timeStep);

	CheckForAndHandleCollisions(colliders, numColliders, particle.position, particle.velocity);

	particle.position += timeStep * (particle.velocity);
}

__global__
void UpdateColliderPositions(ImplicitCollider* colliders, int numColliders, float timestep)
{
	int colliderIdx = blockDim.x * blockIdx.x + threadIdx.x;
	colliders[colliderIdx].center += colliders[colliderIdx].velocity * timestep;
}

__host__
void UpdateParticles(
	Particle* particles, ParticleCache* devParticleCache, ParticleCache* hostParticleCache, int numParticles,
	Grid* grid, Node* nodes, NodeCache* nodeCaches, int numNodes,
	ImplicitCollider* colliders, int numColliders, float timeStep, bool implicitUpdate)
{
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

	// Clear data before update
	cudaMemset(nodes, 0, numNodes * sizeof(Node));
	cudaMemset(nodeCaches, 0, numNodes * sizeof(NodeCache));

	// All dat ParticleCache data
	cudaMemset(hostParticleCache->sigmas, 0, numParticles * sizeof(Matrix3));
	cudaMemset(hostParticleCache->Aps, 0, numParticles * sizeof(Matrix3));
	cudaMemset(hostParticleCache->FeHats, 0, numParticles * sizeof(Matrix3));
	cudaMemset(hostParticleCache->ReHats, 0, numParticles * sizeof(Matrix3));
	cudaMemset(hostParticleCache->SeHats, 0, numParticles * sizeof(Matrix3));
	cudaMemset(hostParticleCache->dFs, 0, numParticles * sizeof(Matrix3));

	const dim3 pBlocks1D((numParticles + THREAD_COUNT - 1) / THREAD_COUNT);
	const dim3 nBlocks1D((numNodes + THREAD_COUNT - 1) / THREAD_COUNT);
	const dim3 threads1D(THREAD_COUNT);
	const dim3 pBlocks2D((numParticles + THREAD_COUNT - 1) / THREAD_COUNT, 64);
	const dim3 threads2D(THREAD_COUNT / 64, 64);

	LAUNCH(UpdateColliderPositions<<<numColliders, 1>>>(colliders, numColliders, timeStep));

	LAUNCH(ComputeSigma<<<pBlocks1D, threads1D>>>(particles, devParticleCache, numParticles, grid));

	LAUNCH(ComputeCellMassVelocityAndForceFast<<<pBlocks2D, threads2D>>>(particles, devParticleCache, numParticles, grid, nodes));

	LAUNCH(UpdateNodeVelocities<<<nBlocks1D, threads1D>>>(nodes, numNodes, timeStep, colliders, numColliders, grid, !implicitUpdate));

	if (implicitUpdate)
	{
		IntegrateNodeForces(particles, devParticleCache, numParticles, grid, nodes, nodeCaches, numNodes, timeStep);
	}

	LAUNCH(UpdateParticlesFromGrid<<<pBlocks1D, threads1D>>>(particles, numParticles, grid, nodes, timeStep, colliders, numColliders));
}

#endif