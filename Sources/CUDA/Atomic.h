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
#include <device_functions.h>

__device__ __forceinline__
void atomicAdd(Vector3* add, const Vector3& toAdd)
{
    atomicAdd(&(add->x), toAdd.x);
    atomicAdd(&(add->y), toAdd.y);
    atomicAdd(&(add->z), toAdd.z);
}

__device__ __forceinline__
void atomicAdd(Vector3* add, const Matrix3& toAdd)
{
    atomicAdd(&(add->data[0]), toAdd[0]);
    atomicAdd(&(add->data[1]), toAdd[1]);
    atomicAdd(&(add->data[2]), toAdd[2]);
    atomicAdd(&(add->data[3]), toAdd[3]);
    atomicAdd(&(add->data[4]), toAdd[4]);
    atomicAdd(&(add->data[5]), toAdd[5]);
    atomicAdd(&(add->data[6]), toAdd[6]);
    atomicAdd(&(add->data[7]), toAdd[7]);
    atomicAdd(&(add->data[8]), toAdd[8]);
}

#endif