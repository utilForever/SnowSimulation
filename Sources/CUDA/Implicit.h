/*************************************************************************
> File Name: Implicit.h
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Implicit functions compatibles with CUDA.
> Created Time: 2018/01/14
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#ifndef SNOW_SIMULATION_IMPLICIT_H
#define SNOW_SIMULATION_IMPLICIT_H

#define CUDA_INCLUDE
#include <Common/Util.h>
#include <CUDA/Atomic.h>
#include <CUDA/Decomposition.h>
#include <CUDA/Helpers.h>
#include <CUDA/Vector.h>
#include <CUDA/Weighting.h>
#include <Geometry/Grid.h>
#include <Simulation/Caches.h>
#include <Simulation/Material.h>
#include <Simulation/Node.h>
#include <Simulation/Particle.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>

constexpr float BETA = 0.5f;
constexpr int MAX_ITERATIONS = 15;
constexpr double RESIDUAL_THRESHOLD = 1e-20;

// Called over particles
__global__
void ComputedF(
    const Particle* particles, ParticleCache* particleCache, int numParticles,
    const Grid* grid, const NodeCache* nodeCaches, NodeCache::Offset uOffset, float dt)
{
    int particleIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (particleIdx >= numParticles)
    {
        return;
    }

    const Particle& particle = particles[particleIdx];
    const glm::ivec3& dim = grid->dim;

    // Compute neighborhood of particle in grid
    Vector3 gridIndex = (particle.position - grid->pos) / grid->h;
    Vector3 gridMax = Vector3::Floor(gridIndex + Vector3(2, 2, 2));
    Vector3 gridMin = Vector3::Ceil(gridIndex - Vector3(2, 2, 2));
    glm::ivec3 maxIndex = glm::clamp(glm::ivec3(gridMax), glm::ivec3(0, 0, 0), dim);
    glm::ivec3 minIndex = glm::clamp(glm::ivec3(gridMin), glm::ivec3(0, 0, 0), dim);

    // Fill dF
    Matrix3 dF(0.0f);
    int rowSize = dim.z + 1;
    int pageSize = (dim.y + 1) * rowSize;

    for (int i = minIndex.x; i <= maxIndex.x; ++i)
    {
        Vector3 d, s;
        d.x = gridIndex.x - i;
        d.x *= (s.x = (d.x < 0) ? -1.f : 1.f);

        int pageOffset = i * pageSize;

        for (int j = minIndex.y; j <= maxIndex.y; ++j)
        {
            d.y = gridIndex.y - j;
            d.y *= (s.y = (d.y < 0) ? -1.f : 1.f);

            int rowOffset = pageOffset + j * rowSize;

            for (int k = minIndex.z; k <= maxIndex.z; ++k)
            {
                d.z = gridIndex.z - k;
                d.z *= (s.z = (d.z < 0) ? -1.f : 1.f);

                Vector3 wg;
                WeightGradient(s, d, wg);

                const NodeCache& nodeCache = nodeCaches[rowOffset + k];
                Vector3 du_j = dt * nodeCache[uOffset];

                dF += Matrix3::OuterProduct(du_j, wg);
            }
        }
    }

    particleCache->dFs[particleIdx] = dF * particle.elasticF;
}

// Currently computed in computedF, we could parallelize this and computedF but not sure what the time benefit would be
__global__
void ComputeFeHat(
    Particle* particles, ParticleCache* particleCache, int numParticles,
    Grid* grid, float dt, Node* nodes)
{
    int particleIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (particleIdx >= numParticles)
    {
        return;
    }

    Particle& particle = particles[particleIdx];
    Vector3 particleGridPos = (particle.position - grid->pos) / grid->h;
    glm::ivec3 min = glm::ivec3(std::ceil(particleGridPos.x - 2), std::ceil(particleGridPos.y - 2), std::ceil(particleGridPos.z - 2));
    glm::ivec3 max = glm::ivec3(std::floor(particleGridPos.x + 2), std::floor(particleGridPos.y + 2), std::floor(particleGridPos.z + 2));
    Matrix3 vGradient(0.0f);

    min = glm::max(glm::ivec3(0.0f), min);
    max = glm::min(grid->dim, max);

    for (int i = min.x; i <= max.x; i++)
    {
        for (int j = min.y; j <= max.y; j++)
        {
            for (int k = min.z; k <= max.z; k++)
            {
                int currIdx = grid->GetGridIndex(i, j, k, grid->dim + 1);
                Node& node = nodes[currIdx];

                Vector3 wg;
                WeightGradient(particleGridPos - Vector3(i, j, k), wg);

                vGradient += Matrix3::OuterProduct(dt * node.velocity, wg);
            }
        }
    }

    Matrix3& FeHat = particleCache->FeHats[particleIdx];
    Matrix3& ReHat = particleCache->ReHats[particleIdx];
    Matrix3& SeHat = particleCache->SeHats[particleIdx];

    FeHat = Matrix3::AddIdentity(vGradient) * particle.elasticF;
    ComputePD(FeHat, ReHat, SeHat);
}

// Computes dR
// FeHat = Re * Se (polar decomposition)
// Re is assumed to be orthogonal
// Se is assumed to be symmetry Positive semi definite
__device__
void ComputedR(const Matrix3& dF, const Matrix3& Se, const Matrix3& Re, Matrix3& dR)
{
    Matrix3 V = Matrix3::MultiplyATB(Re, dF) - Matrix3::MultiplyATB(dF, Re);

    // Solve for compontents of R^T * dR
    // NOTE: remember, column major
    Matrix3 A = Matrix3(
        Se[0] + Se[4], Se[5], -Se[2], 
        Se[5], Se[0] + Se[8], Se[1],
        -Se[2], Se[1], Se[4] + Se[8]);

    Vector3 b(V[3], V[6], V[7]);
    // Should replace this with a linear system solver function
    Vector3 x = Matrix3::Solve(A, b);

    // Fill R^T * dR
    // NOTE: remember, column major
    Matrix3 RTdR = Matrix3(
        0, -x.x, -x.y,
        x.x, 0, -x.z,
        x.y, x.z, 0);

    dR = Re * RTdR;
}

/*
*
* This function involves taking the partial derivative of the cofactor of F
* with respect to each element of F. This process results in a 3x3 block matrix
* where each block is the 3x3 partial derivative for an element of F
*
* Let F = [ a b c
*           d e f
*           g h i ]
*
* Let cofactor(F) = [ ei-hf  gf-di  dh-ge
*                     hc-bi  ai-gc  gb-ah
*                     bf-ec  dc-af  ae-db ]
*
* Then d/da (cofactor(F) = [ 0   0   0
*                            0   i  -h
*                            0  -f   e ]
*
* The other 8 partials will have similar form. See (and run) the code in
* matlab/derivateAdjugateF.m for the full computation as well as to see where
* these seemingly magic values came from.
*
*/
__device__
void ComputedJFInvTrans(const Matrix3& F, const Matrix3& dF, Matrix3& dJFInvTrans)
{
    dJFInvTrans[0] = F[4] * dF[8] - F[5] * dF[7] - F[7] * dF[5] + F[8] * dF[4];
    dJFInvTrans[1] = F[5] * dF[6] - F[3] * dF[8] + F[6] * dF[5] - F[8] * dF[3];
    dJFInvTrans[2] = F[3] * dF[7] - F[4] * dF[6] - F[6] * dF[4] + F[7] * dF[3];
    dJFInvTrans[3] = F[2] * dF[7] - F[1] * dF[8] + F[7] * dF[2] - F[8] * dF[1];
    dJFInvTrans[4] = F[0] * dF[8] - F[2] * dF[6] - F[6] * dF[2] + F[8] * dF[0];
    dJFInvTrans[5] = F[1] * dF[6] - F[0] * dF[7] + F[6] * dF[1] - F[7] * dF[0];
    dJFInvTrans[6] = F[1] * dF[5] - F[2] * dF[4] - F[4] * dF[2] + F[5] * dF[1];
    dJFInvTrans[7] = F[2] * dF[3] - F[0] * dF[5] + F[3] * dF[2] - F[5] * dF[0];
    dJFInvTrans[8] = F[0] * dF[4] - F[1] * dF[3] - F[3] * dF[1] + F[4] * dF[0];
}

// Called over particles
__global__
void ComputeAp(const Particle* particles, ParticleCache* particleCache, int numParticles)
{
    int particleIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (particleIdx >= numParticles)
    {
        return;
    }

    const Particle& particle = particles[particleIdx];
    const Material& material = particle.material;

    const Matrix3& dF = particleCache->dFs[particleIdx];

    // for the sake of making the code look like the math
    const Matrix3& Fp = particle.plasticF;
    const Matrix3& Fe = particleCache->FeHats[particleIdx];
    const Matrix3& Re = particleCache->ReHats[particleIdx];
    const Matrix3& Se = particleCache->SeHats[particleIdx];

    float Jpp = Matrix3::Determinant(Fp);
    float Jep = Matrix3::Determinant(Fe);

    float muFp = material.mu * std::expf(material.xi * (1 - Jpp));
    float lambdaFp = material.lambda * std::expf(material.xi * (1 - Jpp));

    Matrix3 dR;
    ComputedR(dF, Se, Re, dR);

    Matrix3 dJFeInvTrans;
    ComputedJFInvTrans(Fe, dF, dJFeInvTrans);

    Matrix3 JFeInvTrans = Matrix3::Cofactor(Fe);

    particleCache->Aps[particleIdx] = (2 * muFp*(dF - dR) + lambdaFp * JFeInvTrans * Matrix3::InnerProduct(JFeInvTrans, dF) + lambdaFp * (Jep - 1)*dJFeInvTrans);
}

__global__
void ComputedF(
    const Particle* particles, const ParticleCache* particleCache, int numParticles,
    const Grid* grid, NodeCache* nodeCaches)
{
    int particleIdx = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
    if (particleIdx >= numParticles)
    {
        return;
    }

    const Particle& particle = particles[particleIdx];
    Vector3 gridPos = (particle.position - grid->pos) / grid->h;

    glm::ivec3 ijk;
    Grid::GridIndexToIJK(threadIdx.y, glm::ivec3(4, 4, 4), ijk);
    ijk += glm::ivec3(gridPos - 1);

    if (Grid::WithinBoundsInclusive(ijk, glm::ivec3(0, 0, 0), grid->dim))
    {
        Vector3 wg;
        Vector3 nodePos(ijk);
        WeightGradient(gridPos - nodePos, wg);
        
        Vector3 dfJ = -particle.volume * Matrix3::MultiplyABT(particleCache->Aps[particleIdx], particle.elasticF) * wg;

        int gridIndex = Grid::GetGridIndex(ijk, grid->NodeDim());
        atomicAdd(&(nodeCaches[gridIndex].df), dfJ);
    }
}

__global__
void ComputeEuResult(
    const Node* nodes, NodeCache* nodeCaches, int numNodes,
    float dt, NodeCache::Offset uOffset, NodeCache::Offset resultOffset)
{
    int nodeIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (nodeIdx >= numNodes)
    {
        return;
    }

    NodeCache& nodeCache = nodeCaches[nodeIdx];
    float mass = nodes[nodeIdx].mass;
    float scale = (mass > 0.f) ? 1.f / mass : 0.f;
    nodeCache[resultOffset] = nodeCache[uOffset] - BETA * dt * scale * nodeCache.df;
}

__global__
void ZerodF(NodeCache* nodeCaches, int numNodes)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= numNodes)
    {
        return;
    }

    nodeCaches[tid].df = Vector3(0.0f);
}

// Computes the matrix-vector product Eu.
__host__
void ComputeEu(
    const Particle* particles, ParticleCache* particleCache, int numParticles,
    const Grid* grid, const Node* nodes, NodeCache* nodeCaches, int numNodes,
    NodeCache::Offset uOffset, NodeCache::Offset resultOffset, float dt)
{
    const dim3 pBlocks1D((numParticles + THREAD_COUNT - 1) / THREAD_COUNT);
    const dim3 nBlocks1D((numNodes + THREAD_COUNT - 1) / THREAD_COUNT);
    static const dim3 threads1D(THREAD_COUNT);
    const dim3 pBlocks2D((numParticles + THREAD_COUNT - 1) / THREAD_COUNT, 64);
    static const dim3 threads2D(THREAD_COUNT / 64, 64);

    LAUNCH(ComputedF<<<pBlocks1D, threads1D>>>(particles, particleCache, numParticles, grid, nodeCaches, uOffset, dt));

    LAUNCH(ComputeAp<<<pBlocks1D, threads1D>>>(particles, particleCache, numParticles));

    LAUNCH(ZerodF<<<nBlocks1D, threads1D>>>(nodeCaches, numNodes));

    LAUNCH(ComputedF<<<pBlocks2D, threads2D>>>(particles, particleCache, numParticles, grid, nodeCaches));

    LAUNCH(ComputeEuResult<<<nBlocks1D, threads1D>>>(nodes, nodeCaches, numNodes, dt, uOffset, resultOffset));
}

__global__
void InitializeVKernel(const Node* nodes, NodeCache* nodeCaches, int numNodes)
{
    int nodeIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (nodeIdx >= numNodes)
    {
        return;
    }

    nodeCaches[nodeIdx].v = nodes[nodeIdx].velocity;
}

__global__
void InitializeRPKernel(NodeCache* nodeCaches, int numNodes)
{
    int nodeIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (nodeIdx >= numNodes)
    {
        return;
    }

    NodeCache& nodeCache = nodeCaches[nodeIdx];
    nodeCache.r = nodeCache.v - nodeCache.r;
    nodeCache.p = nodeCache.r;
}

__global__
void InitializeApKernel(NodeCache* nodeCaches, int numNodes)
{
    int nodeIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (nodeIdx >= numNodes)
    {
        return;
    }

    NodeCache& nodeCache = nodeCaches[nodeIdx];
    nodeCache.Ap = nodeCache.Ar;
}

__global__
void UpdateVRKernel(NodeCache* nodeCaches, int numNodes, double alpha)
{
    int nodeIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (nodeIdx >= numNodes)
    {
        return;
    }

    NodeCache& nodeCache = nodeCaches[nodeIdx];
    nodeCache.v += alpha * nodeCache.p;
    nodeCache.r -= alpha * nodeCache.Ap;
}

__global__
void UpdatePApResidualKernel(NodeCache* nodeCaches, int numNodes, double beta)
{
    int nodeIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (nodeIdx >= numNodes)
    {
        return;
    }

    NodeCache& nodeCache = nodeCaches[nodeIdx];
    nodeCache.p = nodeCache.r + beta * nodeCache.p;
    nodeCache.Ap = nodeCache.Ar + beta * nodeCache.Ap;
    nodeCache.scratch = static_cast<double>(Vector3::Dot(nodeCache.r, nodeCache.r));
}

__global__
void FinishConjugateResidualKernel(Node* nodes, const NodeCache* nodeCaches, int numNodes)
{
    int nodeIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (nodeIdx >= numNodes)
    {
        return;
    }

    nodes[nodeIdx].velocity = nodeCaches[nodeIdx].v;
    // Update the velocity change. It is assumed to be set as the pre-update velocity
    nodes[nodeIdx].velocityChange = nodes[nodeIdx].velocity - nodes[nodeIdx].velocityChange;
}

__global__
void ScratchReduceKernel(NodeCache* nodeCaches, int numNodes, int reductionSize)
{
    int nodeIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (nodeIdx >= numNodes || nodeIdx + reductionSize >= numNodes)
    {
        return;
    }

    nodeCaches[nodeIdx].scratch += nodeCaches[nodeIdx + reductionSize].scratch;
}

__host__ double ScratchSum(NodeCache* nodeCaches, int numNodes)
{
    static const dim3 blocks((numNodes + THREAD_COUNT - 1) / THREAD_COUNT);
    static const dim3 threads(THREAD_COUNT);
    int steps = static_cast<int>(ceilf(log2f(numNodes)));
    int reductionSize = 1 << (steps - 1);
    
    for (int i = 0; i < steps; i++)
    {
        ScratchReduceKernel<<<blocks, threads>>>(nodeCaches, numNodes, reductionSize);
        reductionSize /= 2;
        cudaDeviceSynchronize();
    }

    double result;
    cudaMemcpy(&result, &(nodeCaches[0].scratch), sizeof(double), cudaMemcpyDeviceToHost);
    return result;
}

__global__
void InnerProductKernel(NodeCache* nodeCaches, int numNodes, NodeCache::Offset uOffset, NodeCache::Offset vOffset)
{
    int nodeIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (nodeIdx >= numNodes)
    {
        return;
    }

    NodeCache& nodeCache = nodeCaches[nodeIdx];
    nodeCache.scratch = static_cast<double>(Vector3::Dot(nodeCache[uOffset], nodeCache[vOffset]));
}

__host__
double InnerProduct(NodeCache* nodeCaches, int numNodes, NodeCache::Offset uOffset, NodeCache::Offset vOffset)
{
    const dim3 blocks((numNodes + THREAD_COUNT - 1) / THREAD_COUNT);
    static const dim3 threads(THREAD_COUNT);

    LAUNCH(InnerProductKernel<<<blocks, threads>>>(nodeCaches, numNodes, uOffset, vOffset));
    
    return ScratchSum(nodeCaches, numNodes);
}

__host__
void IntegrateNodeForces(
    Particle* particles, ParticleCache* particleCache, int numParticles,
    Grid* grid, Node* nodes, NodeCache* nodeCaches, int numNodes, float dt)
{
    const dim3 blocks((numNodes + THREAD_COUNT - 1) / THREAD_COUNT);
    static const dim3 threads(THREAD_COUNT);

    // No need to sync because it can run in parallel with other kernels
    ComputeFeHat<<<(numParticles + THREAD_COUNT - 1) / THREAD_COUNT, THREAD_COUNT>>>(particles, particleCache, numParticles, grid, dt, nodes);

    // Initialize conjugate residual method
    LAUNCH(InitializeVKernel<<<blocks, threads>>>(nodes, nodeCaches, numNodes));
    ComputeEu(particles, particleCache, numParticles, grid, nodes, nodeCaches, numNodes, NodeCache::Offset::V, NodeCache::Offset::R, dt);
    LAUNCH(InitializeRPKernel<<<blocks, threads>>>(nodeCaches, numNodes));
    ComputeEu(particles, particleCache, numParticles, grid, nodes, nodeCaches, numNodes, NodeCache::Offset::R, NodeCache::Offset::AR, dt);
    LAUNCH(InitializeApKernel<<<blocks, threads>>>(nodeCaches, numNodes));

    int k = 0;
    float residual;
    do
    {
        double alphaNum = InnerProduct(nodeCaches, numNodes, NodeCache::Offset::R, NodeCache::Offset::AR);
        double alphaDen = InnerProduct(nodeCaches, numNodes, NodeCache::Offset::AP, NodeCache::Offset::AP);
        double alpha = (fabsf(alphaDen) > 0.f) ? alphaNum / alphaDen : 0.f;
        double betaDen = alphaNum;

        LAUNCH(UpdateVRKernel<<<blocks, threads>>>(nodeCaches, numNodes, alpha));
        ComputeEu(particles, particleCache, numParticles, grid, nodes, nodeCaches, numNodes, NodeCache::Offset::R, NodeCache::Offset::AR, dt);
        
        double betaNum = InnerProduct(nodeCaches, numNodes, NodeCache::Offset::R, NodeCache::Offset::AR);
        double beta = (fabsf(betaDen) > 0.f) ? betaNum / betaDen : 0.f;

        LAUNCH(UpdatePApResidualKernel<<<blocks, threads>>>(nodeCaches, numNodes, beta));
        residual = ScratchSum(nodeCaches, numNodes);

        LOG("k = %3d, rAr = %10g, alpha = %10g, beta = %10g, r = %g", k, alphaNum, alpha, beta, residual);

    } while (++k < MAX_ITERATIONS && residual > RESIDUAL_THRESHOLD);

    LAUNCH(FinishConjugateResidualKernel<<<blocks, threads>>>(nodes, nodeCaches, numNodes));
}

#endif