/*************************************************************************
> File Name: Collider.h
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Collider functions compatibles with CUDA.
> Created Time: 2018/01/13
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#ifndef SNOW_SIMULATION_COLLIDER_H
#define SNOW_SIMULATION_COLLIDER_H

#include <CUDA/Matrix.h>
#include <CUDA/Vector.h>
#include <Simulation/ImplicitCollider.h>

#ifndef GLM_FORCE_RADIANS
#define GLM_FORCE_RADIANS
#endif

#include <glm/geometric.hpp>

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#include <functional>

// IsColliding functions
// using IsCollidingFunc = std::function<bool(const ImplicitCollider& collider, const Vector3& position)>;
typedef bool(*IsCollidingFunc)(const ImplicitCollider& collider, const Vector3& position);

// A collision occurs when the point is on the OTHER side of the normal
__device__
inline bool IsCollidingHalfPlane(const Vector3& planePoint, const Vector3& planeNormal, const Vector3& position)
{
	Vector3 vecToPoint = position - planePoint;
	return (Vector3::Dot(vecToPoint, planeNormal) <= 0);
}

// Defines a half plane such that collider.center is a point on the plane,
// and collider.param is the normal to the plane.
__device__
inline bool IsCollidingHalfPlaneImplicit(const ImplicitCollider& collider, const Vector3& position)
{
	return IsCollidingHalfPlane(collider.center, collider.param, position);
}

// Defines a sphere such that collider.center is the center of the sphere,
// and collider.param.x is the radius.
__device__
inline bool IsCollidingSphereImplicit(const ImplicitCollider& collider, const Vector3& position)
{
	float radius = collider.param.x;
	return (Vector3::Length(position - collider.center) <= radius);
}

// array of colliding functions. isCollidingFunctions[collider.type] will be the correct function
__device__
IsCollidingFunc isCollidingFunctions[2] =
{
	IsCollidingHalfPlaneImplicit,
	IsCollidingSphereImplicit
};

// General purpose function for handling colliders
__device__
inline bool IsColliding(const ImplicitCollider& collider, const Vector3& position)
{
	return isCollidingFunctions[static_cast<int>(collider.type)](collider, position);
}

// colliderNormal functions
// Returns the (normalized) normal of the collider at the position.
// Note: this function does NOT check that there is a collision at this point, and behavior is undefined if there is not.
// using ColliderNormalFunc = std::function<void(const ImplicitCollider& collider, const Vector3& position, Vector3& normal)>;
typedef void(*ColliderNormalFunc)(const ImplicitCollider& collider, const Vector3& position, Vector3& normal);

__device__
inline void ColliderNormalSphere(const ImplicitCollider& collider, const Vector3& position, Vector3& normal)
{
	normal = Vector3::Normalize(position - collider.center);
}

__device__
inline void ColliderNormalHalfPlane(const ImplicitCollider& collider, const Vector3& position, Vector3& normal)
{
	//The half planes normal is stored in collider.param
	normal = collider.param;
}

//array of colliderNormal functions. colliderNormalFunctions[collider.type] will be the correct function
__device__ ColliderNormalFunc colliderNormalFunctions[2] =
{
	ColliderNormalHalfPlane,
	ColliderNormalSphere
};

__device__
inline void ColliderNormal(const ImplicitCollider& collider, const Vector3& position, Vector3& normal)
{
	colliderNormalFunctions[static_cast<int>(collider.type)](collider, position, normal);
}

__device__
inline void CheckForAndHandleCollisions(const ImplicitCollider* colliders, int numColliders, const Vector3& position, Vector3& velocity)
{
	for (int i = 0; i < numColliders; ++i)
	{
		const ImplicitCollider& collider = colliders[i];
	 
		if (IsColliding(collider, position))
		{
			Vector3 vRel = velocity - collider.velocity;
			Vector3 normal;

			ColliderNormal(collider, position, normal);
			
			// Bodies are not separating and a collision must be applied
			float vn = Vector3::Dot(vRel, normal);
			if (vn < 0)
			{
				Vector3 vt = vRel - normal * vn;
				float magVt = Vector3::Length(vt);

				// tangential velocity not enough to overcome force of friction
				if (magVt <= -collider.coeffFriction * vn)
				{ 
					vRel = Vector3(0.0f);
				}
				else
				{
					vRel = (1 + collider.coeffFriction * vn / magVt) * vt;
				}
			}

			velocity = vRel + collider.velocity;
		}
	}
}

#endif