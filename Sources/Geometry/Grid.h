/*************************************************************************
> File Name: Grid.h
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Grid geometry of snow simulation.
> Created Time: 2017/12/31
> Copyright (c) 2017, Chan-Ho Chris Ohk
*************************************************************************/
#ifndef SNOW_SIMULATION_GRID_H
#define SNOW_SIMULATION_GRID_H

#ifndef FUNC
#ifdef CUDA_INCLUDE
#define FUNC __device__ __host__ __forceinline__
#else
#define FUNC inline
#endif
#endif

#ifndef GLM_FORCE_RADIANS
#define GLM_FORCE_RADIANS
#endif

#include <CUDA/Vector.h>

#include <glm/vec3.hpp>

struct Grid
{
    glm::ivec3 dim;
    Vec3 pos;
    float h;

    FUNC Grid() : dim(0, 0, 0), pos(0, 0, 0), h(0.f)
    {
        // Do nothing
    }

    FUNC Grid(const Grid& grid) : dim(grid.dim), pos(grid.pos), h(grid.h)
    {
        // Do nothing
    }

    FUNC int GetNumOfCells() const
    {
        return dim.x * dim.y * dim.z;
    }

    FUNC bool IsEmpty() const
    {
        return dim.x * dim.y * dim.z == 0;
    }

    FUNC glm::ivec3 NodeDim() const
    {
        return dim + glm::ivec3(1, 1, 1);
    }

    FUNC int GetNumOfNodes() const
    {
        return (dim.x + 1) * (dim.y + 1) * (dim.z + 1);
    }

    FUNC int GetIndex(int i, int j, int k) const
    {
        return i * (dim.y * dim.z) + j * (dim.z) + k;
    }

    FUNC static void GridIndexToIJK(int idx, int& i, int& j, int& k, const glm::ivec3& nodeDim)
    {
        i = idx / (nodeDim.y * nodeDim.z);
        idx = idx % (nodeDim.y * nodeDim.z);
        j = idx / nodeDim.z;
        k = idx % nodeDim.z;
    }

    FUNC static void GridIndexToIJK(int idx, const glm::ivec3& nodeDim, glm::ivec3& ijk)
    {
        ijk.x = idx / (nodeDim.y * nodeDim.z);
        idx = idx % (nodeDim.y * nodeDim.z);
        ijk.y = idx / nodeDim.z;
        ijk.z = idx % nodeDim.z;
    }

    FUNC static int GetGridIndex(int i, int j, int k, const glm::ivec3& nodeDim)
    {
        return i * (nodeDim.y * nodeDim.z) + j * nodeDim.z + k;
    }

    FUNC static int GetGridIndex(const glm::ivec3& ijk, const glm::ivec3& nodeDim)
    {
        return ijk.x * (nodeDim.y * nodeDim.z) + ijk.y * nodeDim.z + ijk.z;
    }

    FUNC static bool WithinBoundsInclusive(const float& v, const float& min, const float& max)
    {
        return v >= min && v <= max;
    }

    FUNC static bool WithinBoundsInclusive(const glm::ivec3& v, const glm::ivec3& min, const glm::ivec3& max)
    {
        return WithinBoundsInclusive(v.x, min.x, max.x) && WithinBoundsInclusive(v.y, min.y, max.y) && WithinBoundsInclusive(v.z, min.z, max.z);
    }
};

#endif