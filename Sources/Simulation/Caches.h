/*************************************************************************
> File Name: NodeCache.h
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Node and particle cache structure of snow simulation.
> Created Time: 2018/01/04
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#ifndef SNOW_SIMULATION_CACHES_H
#define SNOW_SIMULATION_CACHES_H

#include <CUDA/Matrix.h>

#include <cuda.h>
#include <cuda_runtime.h>

struct NodeCache
{
	enum class Offset
	{
		R, AR, P, AP, V, DF
	};

	// Data used by Conjugate Residual Method
	Vector3 r;
	Vector3 Ar;
	Vector3 p;
	Vector3 Ap;
	Vector3 v;
	Vector3 df;
	double scratch;
	
	__host__ __device__
	Vector3& operator[](Offset i)
	{
		switch (i)
		{
		case Offset::R:     return r;
		case Offset::AR:    return Ar;
		case Offset::P:     return p;
		case Offset::AP:    return Ap;
		case Offset::V:     return v;
		case Offset::DF:    return df;
		}
		return r;
	}

	__host__ __device__
	Vector3 operator[](Offset i) const
	{
		switch (i)
		{
		case Offset::R:     return r;
		case Offset::AR:    return Ar;
		case Offset::P:     return p;
		case Offset::AP:    return Ap;
		case Offset::V:     return v;
		case Offset::DF:    return df;
		}

		return r;
	}
};

struct ParticleCache
{
	// Data used during initial node computations
	Matrix3* sigmas;

	// Data used during implicit node velocity update
	Matrix3* Aps;
	Matrix3* FeHats;
	Matrix3* ReHats;
	Matrix3* SeHats;
	Matrix3* dFs;
};

#endif