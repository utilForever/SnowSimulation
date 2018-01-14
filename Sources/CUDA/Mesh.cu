/*************************************************************************
> File Name: Mesh.cu
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Mesh CUDA functions.
> Created Time: 2018/01/14
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#ifndef SNOW_SIMULATION_MESH_CU
#define SNOW_SIMULATION_MESH_CU

#define CUDA_INCLUDE
#include <Common/Math.h>
#include <Common/Util.h>
#include <CUDA/Functions.h>
#include <CUDA/Helpers.h>
#include <CUDA/Noise.h>
#include <CUDA/SnowTypes.h>
#include <CUDA/Vector.h>
#include <Geometry/BBox.h>
#include <Geometry/Grid.h>
#include <Simulation/Particle.h>

#include <Windows.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <curand.h>
#include <curand_kernel.h>

#include <glm/gtc/random.hpp>

struct Tri
{
    Vector3 v0, n0;
    Vector3 v1, n1;
    Vector3 v2, n2;
};

// Moller, T, and Trumbore, B. Fast, Minimum Storage Ray/Triangle Intersection.
__device__
int IntersectTri(
    const Vector3& v1, const Vector3& v2, const Vector3& v3,
    const Vector3& O, const Vector3& D, float& t)
{
    Vector3 e1 = v2 - v1;
    Vector3 e2 = v3 - v1;
    Vector3 P = Vector3::Cross(D, e2);

    float det = Vector3::Dot(e1, P);
    if (det > -1e-8 && det < 1e-8)
    {
        return 0;
    }

    float invDet = 1.f / det;
    Vector3 T = O - v1;

    float u = Vector3::Dot(T, P) * invDet;
    if (u < 0.f || u > 1.f)
    {
        return 0;
    }

    Vector3 Q = Vector3::Cross(T, e1);

    float v = Vector3::Dot(D, Q)*invDet;
    if (v < 0.f || u + v  > 1.f)
    {
        return 0;
    }

    t = Vector3::Dot(e2, Q) * invDet;
    // ray intersection
    if (t > 1e-8)
    { 
        return 1;
    }
    
    // No hit, no win
    return 0;
}

__global__
void VoxelizeMeshKernel(Tri* tris, int triCount, Grid grid, bool* flags)
{
    const glm::ivec3& dim = grid.dim;

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    if (x >= dim.x)
    {
        return;
    }
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (y >= dim.y)
    {
        return;
    }

    // Shoot ray in z-direction
    Vector3 origin = grid.pos + grid.h * Vector3(x + 0.5f, y + 0.5f, 0.f);
    Vector3 direction = Vector3(0.f, 0.f, 1.f);

    // Flag surface-intersecting voxels
    float t;
    int xyOffset = x * dim.y*dim.z + y * dim.z, z;
    for (int i = 0; i < triCount; ++i)
    {
        const Tri& tri = tris[i];
        if (IntersectTri(tri.v0, tri.v1, tri.v2, origin, direction, t))
        {
            z = static_cast<int>(t / grid.h);
            flags[xyOffset + z] = true;
        }
    }

    // Scanline to fill inner voxels
    int end = xyOffset + dim.z, zz;
    for (int z = xyOffset; z < end; ++z)
    {
        if (flags[z])
        {
            do
            {
                z++;
            } while (flags[z] && z < end);

            zz = z;

            do
            {
                zz++;
            } while (!flags[zz] && zz < end);
            
            if (zz < end - 1)
            {
                for (int i = z; i < zz; ++i)
                {
                    flags[i] = true;
                }

                z = zz;
            }
            else
            {
                break;
            }
        }
    }
}

__global__
void InitReduction(bool* flags, int voxelCount, int* reduction, int reductionSize)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= reductionSize)
    {
        return;
    }

    reduction[tid] = (tid < voxelCount) ? flags[tid] : 0;
}

__global__
void Reduce(int* reduction, int size)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= size)
    {
        return;
    }

    reduction[tid] += reduction[tid + size];
}

__global__
void FillMeshVoxelsKernel(curandState* states, unsigned int seed, Grid grid, bool* flags, Particle* particles, float particleMass, int particleCount)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= particleCount)
    {
        return;
    }

    curandState& localState = states[tid];
    curand_init(seed, tid, 0, &localState);

    const glm::ivec3& dim = grid.dim;

    // Rejection sample
    unsigned int i;
    unsigned int voxelCount = dim.x * dim.y * dim.z;
    do
    {
        i = curand(&localState) % voxelCount;
    } while (!flags[i]);

    // Get 3D voxel index
    unsigned int x = i / (dim.y * dim.z);
    unsigned int y = (i - x * dim.y * dim.z) / dim.z;
    unsigned int z = i - y * dim.z - x * dim.y * dim.z;

    // Generate random point in voxel cube
    Vector3 r = Vector3(curand_uniform(&localState), curand_uniform(&localState), curand_uniform(&localState));
    Vector3 min = grid.pos + grid.h * Vector3(x, y, z);
    Vector3 max = min + Vector3(grid.h, grid.h, grid.h);

    Particle particle;
    particle.mass = particleMass;
    particle.position = min + r * (max - min);
    particle.velocity = Vector3(0, -1, 0);
    particle.material = Material();
    particles[tid] = particle;
}

void FillMesh(cudaGraphicsResource** resource, int triCount, const Grid& grid, Particle* particles, int particleCount, float targetDensity, int materialPreset)
{
    // Get mesh data
    cudaGraphicsMapResources(1, resource, nullptr);
    Tri* devTris;
    size_t size;
    cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&devTris), &size, *resource);

    // Voxelize mesh
    int x = grid.dim.x > 16 ? std::max(1, std::min(16, grid.dim.x / 8)) : 1;
    int y = grid.dim.y > 16 ? std::max(1, std::min(16, grid.dim.y / 8)) : 1;
    dim3 blocks((grid.dim.x + x - 1) / x, (grid.dim.y + y - 1) / y), threads(x, y);
    int voxelCount = grid.dim.x * grid.dim.y * grid.dim.z;
    bool* devFlags;
    cudaMalloc(reinterpret_cast<void**>(&devFlags), voxelCount * sizeof(bool));
    cudaMemset(static_cast<void*>(devFlags), 0, voxelCount * sizeof(bool));
    VoxelizeMeshKernel<<<blocks, threads>>>(devTris, triCount, grid, devFlags);
    cudaDeviceSynchronize();

    int powerOfTwo = static_cast<int>(log2f(voxelCount) + 1);
    int reductionSize = 1 << powerOfTwo;
    int* devReduction;
    cudaMalloc(reinterpret_cast<void**>(&devReduction), reductionSize * sizeof(int));
    InitReduction<<<(reductionSize + 511) / 512, 512>>>(devFlags, voxelCount, devReduction, reductionSize);
    cudaDeviceSynchronize();

    for (int i = 0; i < powerOfTwo - 1; ++i)
    {
        int size = 1 << (powerOfTwo - i - 1);
        Reduce<<<(size + 511) / 512, 512>>>(devReduction, size);
        cudaDeviceSynchronize();
    }

    int count;
    cudaMemcpy(&count, devReduction, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(devReduction);

    float volume = count * grid.h * grid.h * grid.h;
    float particleMass = targetDensity * volume / particleCount;
    LOG("Average %.2f particles per grid cell.", float(particleCount) / count);
    LOG("Target Density: %.1f kg/m3 -> Particle Mass: %g kg", targetDensity, particleMass);
    
    // Randomly fill mesh voxels and copy back resulting particles
    curandState* devStates;
    cudaMalloc(&devStates, particleCount * sizeof(curandState));
    Particle* devParticles;
    cudaMalloc(reinterpret_cast<void**>(&devParticles), particleCount * sizeof(Particle));
    FillMeshVoxelsKernel<<<(particleCount + 511) / 512, 512>>>(devStates, time(nullptr), grid, devFlags, devParticles, particleMass, particleCount);
    cudaDeviceSynchronize();

    switch (materialPreset)
    {
    case 0:
        break;
    case 1:
        LAUNCH(ApplyChunky<<<(particleCount + 511) / 512, 512>>>(devParticles, particleCount));
        LOG("Chunky applied");
        break;
    default:
        break;
    }

    cudaMemcpy(particles, devParticles, particleCount * sizeof(Particle), cudaMemcpyDeviceToHost);

    cudaFree(devFlags);
    cudaFree(devStates);
    cudaGraphicsUnmapResources(1, resource, nullptr);
}

#endif