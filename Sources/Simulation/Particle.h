/*************************************************************************
> File Name: Particle.h
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Particle structure of snow simulation.
> Created Time: 2018/01/01
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#ifndef SNOW_SIMULATION_PARTICLE_H
#define SNOW_SIMULATION_PARTICLE_H

#include <CUDA/Matrix.h>
#include <Simulation/Material.h>

#include <cuda.h>
#include <cuda_runtime.h>

struct Particle
{
    Vector3 position;
    Vector3 velocity;
    float mass;
    float volume;
    Matrix3 elasticF;
    Matrix3 plasticF;
    Material material;

    __host__ __device__
    Particle() :
        position(Vector3(0.f, 0.f, 0.f)),
        velocity(Vector3(0.f, 0.f, 0.f)),
        mass(1e-6), volume(1e-9),
        elasticF(Matrix3(1.f)),
        plasticF(Matrix3(1.f)),
        material(Material())
    {
        // Do nothing
    }
};

#endif