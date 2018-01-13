/*************************************************************************
> File Name: Snow.cu
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Snow CUDA functions.
> Created Time: 2018/01/13
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#ifndef SNOW_SIMULATION_SNOW_CU
#define SNOW_SIMULATION_SNOW_CU

#define CUDA_INCLUDE
#include <CUDA/Functions.h>
#include <Simulation/Particle.h>

#include <Windows.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#ifndef GLM_FORCE_RADIANS
#define GLM_FORCE_RADIANS
#endif

#include <glm/geometric.hpp>

void RegisterVBO(cudaGraphicsResource** resource, GLuint vbo)
{
    cudaGraphicsGLRegisterBuffer(resource, vbo, cudaGraphicsMapFlagsNone);
}

void UnregisterVBO(cudaGraphicsResource *resource)
{
    cudaGraphicsUnregisterResource(resource);
}

#endif