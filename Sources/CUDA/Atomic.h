/*************************************************************************
> File Name: Atomic.h
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Atomic functions compatibles with CUDA.
> Created Time: 2018/01/13
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#ifndef SNOW_SIMULATION_ATOMIC_H
#define SNOW_SIMULATION_ATOMIC_H

#define CUDA_INCLUDE
#include <CUDA/Matrix.h>
#include <CUDA/Vector.h>

#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__
void AtomicAdd(Vector3* add, const Vector3& toAdd)
{
    AtomicAdd(&(add->x), toAdd.x);
    AtomicAdd(&(add->y), toAdd.y);
    AtomicAdd(&(add->z), toAdd.z);
}

__device__ __forceinline__
void AtomicAdd(Vector3* add, const Matrix3& toAdd)
{
    AtomicAdd(&(add->data[0]), toAdd[0]);
    AtomicAdd(&(add->data[1]), toAdd[1]);
    AtomicAdd(&(add->data[2]), toAdd[2]);
    AtomicAdd(&(add->data[3]), toAdd[3]);
    AtomicAdd(&(add->data[4]), toAdd[4]);
    AtomicAdd(&(add->data[5]), toAdd[5]);
    AtomicAdd(&(add->data[6]), toAdd[6]);
    AtomicAdd(&(add->data[7]), toAdd[7]);
    AtomicAdd(&(add->data[8]), toAdd[8]);
}

#endif