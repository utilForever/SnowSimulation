/*************************************************************************
> File Name: Quaternion.h
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Quaternion type compatibles with CUDA.
> Created Time: 2018/01/01
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#ifndef SNOW_SIMULATION_QUATERNION_H
#define SNOW_SIMULATION_QUATERNION_H

#include <Common/Math.h>

#include <cuda.h>
#include <cuda_runtime.h>

struct Quaternion
{
	union
	{
		float data[4];
		struct
		{
			float x;
			float y;
			float z;
			float w;
		};
	};

	__host__ __device__ __forceinline__
	Quaternion()
	{
		x = 0.f;
		y = 0.f;
		z = 0.f;
		w = 1.f;
	}

	__host__ __device__ __forceinline__
	Quaternion(float ww, float xx, float yy, float zz)
	{
		x = xx;
		y = yy;
		z = zz;
		w = ww;
	}

	__host__ __device__ __forceinline__
	Quaternion(const Quaternion& q)
	{
		x = q.x;
		y = q.y;
		z = q.z;
		w = q.w;
	}

	__host__ __device__ __forceinline__
	Quaternion& operator=(const Quaternion& q)
	{
		x = q.x;
		y = q.y;
		z = q.z;
		w = q.w;
		
		return *this;
	}

	__host__ __device__ __forceinline__
	float& operator[](int i)
	{
		return data[i];
	}

	__host__ __device__ __forceinline__
	float operator[](int i) const
	{
		return data[i];
	}

	__host__ __device__ __forceinline__
	Quaternion& operator*=(float f)
	{
		x *= f;
		y *= f;
		z *= f;
		w *= f;
		
		return *this;
	}

	__host__ __device__ __forceinline__
	Quaternion operator*(float f) const
	{
		return Quaternion(w * f, x * f, y * f, z * f);
	}

	__host__ __device__ __forceinline__
	Quaternion& operator*=(const Quaternion& q)
	{
		Quaternion p(*this);
	   
		w = p.w * q.w - p.x * q.x - p.y * q.y - p.z * q.z;
		x = p.w * q.x + p.x * q.w + p.y * q.z - p.z * q.y;
		y = p.w * q.y + p.y * q.w + p.z * q.x - p.x * q.z;
		z = p.w * q.z + p.z * q.w + p.x * q.y - p.y * q.x;

		return *this;
	}

	__host__ __device__ __forceinline__
	Quaternion operator*(const Quaternion& q) const
	{
		Quaternion p;

		p.w = w * q.w - x * q.x - y * q.y - z * q.z;
		p.x = w * q.x + x * q.w + y * q.z - z * q.y;
		p.y = w * q.y + y * q.w + z * q.x - x * q.z;
		p.z = w * q.z + z * q.w + x * q.y - y * q.x;
		
		return p;
	}
};

#endif