/*************************************************************************
> File Name: Weighting.h
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Weighting functions compatibles with CUDA.
> Created Time: 2018/01/14
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#ifndef SNOW_SIMULATION_WEIGHTING_H
#define SNOW_SIMULATION_WEIGHTING_H

#include <CUDA/Vector.h>

#include <cuda.h>
#include <cuda_runtime.h>

// 1D B-spline falloff
// d is the distance from the point to the node center,
// normalized by h such that particles < 1 grid cell away
// will have 0 < d < 1, particles > 1 and < 2 grid cells away will
// still get some weight, and any particles further than that get
// weight = 0
__host__ __device__ __forceinline__
inline float GetN(float d)
{
	if (d >= 0 && d < 1)
	{
		return 0.5f * d * d * d - d * d + (2.f / 3.f);
	}
	
	if (d >= 1 && d < 2)
	{
		return (-1.f / 6.f) * d * d * d + d * d - 2 * d + (4.f / 3.f);
	}

	return 0.0f;
}

// Sets w = interpolation weights (w_ip)
// input is dx because we'd rather pre-compute abs outside so we can re-use again
// in the weightGradient function.
// by paper notation, w_ip = N_{i}^{h}(p) = N((xp-ih)/h)N((yp-jh)/h)N((zp-kh)/h)
__host__ __device__ __forceinline__
void Weight(Vector3& dx, float h, float& w)
{
	w = GetN(dx.x / h) * GetN(dx.y / h) * GetN(dx.z / h);
}

// Same as above, but dx is already normalized with h
__host__ __device__ __forceinline__
void Weight(Vector3& dx, float& w)
{
	w = GetN(dx.x) * GetN(dx.y) * GetN(dx.z);
}

__host__ __device__ __forceinline__
float Weight(Vector3& dx)
{
	return GetN(dx.x) * GetN(dx.y) * GetN(dx.z);
}

// derivative of N with respect to d
__host__ __device__ __forceinline__
inline float GetNd(float d)
{
	if (d >= 0 && d < 1)
	{
		return 1.5f * d * d - 2 * d;
	}

	if (d >= 1 && d < 2)
	{
		return -.5f * d * d + 2 * d - 2;
	}

	return 0.0f;
}

// returns gradient of interpolation weights  \grad{w_ip}
// xp = sign( distance from grid node to particle )
// dx = abs( distance from grid node to particle )
__host__ __device__ __forceinline__
void WeightGradient(const Vector3& sdx, const Vector3& dx, float h, Vector3& wg)
{
	const Vector3 dx_h = dx / h;
	const Vector3 N = Vector3(GetN(dx_h.x), GetN(dx_h.y), GetN(dx_h.z));
	const Vector3 Nx = sdx * Vector3(GetNd(dx_h.x), GetNd(dx_h.y), GetNd(dx_h.z));

	wg.x = Nx.x * N.y * N.z;
	wg.y = N.x  * Nx.y* N.z;
	wg.z = N.x  * N.y * Nx.z;
}

// Same as above, but dx is already normalized with h
__host__ __device__ __forceinline__
void WeightGradient(const Vector3& sdx, const Vector3& dx, Vector3& wg)
{
	const Vector3 N = Vector3(GetN(dx.x), GetN(dx.y), GetN(dx.z));
	const Vector3 Nx = sdx * Vector3(GetNd(dx.x), GetNd(dx.y), GetNd(dx.z));

	wg.x = Nx.x * N.y * N.z;
	wg.y = N.x  * Nx.y* N.z;
	wg.z = N.x  * N.y * Nx.z;
}

// Same as above, but dx is not already absolute-valued
__host__ __device__ __forceinline__
void WeightGradient(const Vector3& dx, Vector3& wg)
{
	const Vector3 sdx = Vector3::Sign(dx);
	const Vector3 adx = Vector3::Abs(dx);
	const Vector3 N = Vector3(GetN(adx.x), GetN(adx.y), GetN(adx.z));
	const Vector3 Nx = sdx * Vector3(GetNd(adx.x), GetNd(adx.y), GetNd(adx.z));

	wg.x = Nx.x * N.y * N.z;
	wg.y = N.x  * Nx.y* N.z;
	wg.z = N.x  * N.y * Nx.z;
}

// returns weight and gradient of weight, avoiding duplicate computations if applicable
__host__ __device__ __forceinline__
void WeightAndGradient(const Vector3& sdx, const Vector3& dx, float h, float& w, Vector3& wg)
{
	const Vector3 dx_h = dx / h;
	const Vector3 N = Vector3(GetN(dx_h.x), GetN(dx_h.y), GetN(dx_h.z));
	const Vector3 Nx = sdx * Vector3(GetNd(dx_h.x), GetNd(dx_h.y), GetNd(dx_h.z));

	w = N.x * N.y * N.z;
	wg.x = Nx.x * N.y * N.z;
	wg.y = N.x  * Nx.y* N.z;
	wg.z = N.x  * N.y * Nx.z;
}

// Same as above, but dx is already normalized with h
__host__ __device__ __forceinline__
void WeightAndGradient(const Vector3& sdx, const Vector3& dx, float& w, Vector3& wg)
{
	const Vector3 N = Vector3(GetN(dx.x), GetN(dx.y), GetN(dx.z));
	const Vector3 Nx = sdx * Vector3(GetNd(dx.x), GetNd(dx.y), GetNd(dx.z));

	w = N.x * N.y * N.z;
	wg.x = Nx.x * N.y * N.z;
	wg.y = N.x  * Nx.y* N.z;
	wg.z = N.x  * N.y * Nx.z;
}

// Same as above, but dx is not already absolute-valued
__host__ __device__ __forceinline__
void WeightAndGradient(const Vector3& dx, float& w, Vector3& wg)
{
	const Vector3 sdx = Vector3::Sign(dx);
	const Vector3 adx = Vector3::Abs(dx);
	const Vector3 N = Vector3(GetN(adx.x), GetN(adx.y), GetN(adx.z));
	const Vector3 Nx = sdx * Vector3(GetNd(adx.x), GetNd(adx.y), GetNd(adx.z));

	w = N.x * N.y * N.z;
	wg.x = Nx.x * N.y * N.z;
	wg.y = N.x  * Nx.y* N.z;
	wg.z = N.x  * N.y * Nx.z;
}

#endif