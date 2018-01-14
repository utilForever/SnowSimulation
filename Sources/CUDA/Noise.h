/*************************************************************************
> File Name: Noise.h
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Noise functions compatibles with CUDA.
> Created Time: 2018/01/14
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#ifndef SNOW_SIMULATION_NOISE_H
#define SNOW_SIMULATION_NOISE_H

#include <Common/Math.h>
#include <CUDA/Vector.h>

#include <cuda.h>
#include <math.h>

// fractional component of a number
__device__ __forceinline__
float Fract(float x)
{
	return x - floor(x);
}

// Noise functions from IQ.
// https://www.shadertoy.com/view/lsj3zy
__device__ __forceinline__
float Hash(float n)
{
	return Fract(sin(n) * 43758.5453123);
}

__device__ __forceinline__
Vector3 Fract(const Vector3& v)
{
	return v - Vector3::Floor(v);
}

__device__ __forceinline__
float Mix(float x, float y, float a)
{
	return (1.f - a) * x + a * y;
}

__device__ __forceinline__
float Noise3(Vector3 x)
{
	Vector3 p = Vector3::Floor(x);
	Vector3 f = Fract(x);
	f = f * f * (Vector3(3.0) - 2.0 * f);

	float n = p.x + p.y * 157.0 + 113.0 * p.z;
	return
		Mix(Mix(Mix(Hash(n + 0.0), Hash(n + 1.0), f.x),
		Mix(Hash(n + 157.0), Hash(n + 158.0), f.x), f.y),
		Mix(Mix(Hash(n + 113.0), Hash(n + 114.0), f.x),
		Mix(Hash(n + 270.0), Hash(n + 271.0), f.x), f.y), f.z);
}

// 3D fractal brownian motion - https://www.shadertoy.com/view/lsj3zy
__device__ __forceinline__
float FBM3(Vector3 p)
{
	float f = 0.0, x;

	// level of detail
	for (int i = 1; i <= 6; ++i)
	{
		x = 1 << i;
		f += (Noise3(p * x) - 0.5) / x;
	}

	// returns range between 0,1
	return (f + .3) * 1.6667;
}

__host__ __device__ __forceinline__
float Halton(int index, int base)
{
	// base is a fixed prime number for each sequence
	float result = 0.f;
	float f = 1 / float(base);
	int i = index;

	while (i > 0)
	{
		result = result + f * (i % base);
		i /= base;
		f /= base;
	}

	return result;
}

#endif