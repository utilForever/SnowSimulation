/*************************************************************************
> File Name: ImplicitCollider.h
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Implicit collider structure of snow simulation.
> Created Time: 2018/01/01
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#ifndef SNOW_SIMULATION_IMPLICIT_COLLIDER_H
#define SNOW_SIMULATION_IMPLICIT_COLLIDER_H

#include <CUDA/Vector.h>

#include <cuda.h>
#include <cuda_runtime.h>

#ifndef GLM_FORCE_RADIANS
#define GLM_FORCE_RADIANS
#endif
#include <glm/mat4x4.hpp>
#include <glm/gtc/type_ptr.hpp>

/*
	For the sake of supporting multiple implicit colliders in CUDA, we define an enum for the type of collider
	and use the ImplicitCollider.param to specify the collider once the type is known. Most simple implicit shapes
	can be parameterized using at most 3 parameters. For instance, a half-plane is a point (ImplicitCollider.center)
	and a normal (ImplicitCollider.param). A sphere is a center (ImplicitCollider.center) and a radius (ImplicitCollider.param.x)
*/

enum class ColliderType
{
	HALF_PLANE = 0,
	SPHERE = 1
};

struct ImplicitCollider
{
	ColliderType type;
	Vector3 center;
	Vector3 param;
	Vector3 velocity;
	float coeffFriction;

	__host__ __device__
	ImplicitCollider() : type(ColliderType::HALF_PLANE), center(0, 0, 0), param(0, 1, 0),  velocity(0, 0, 0), coeffFriction(0.1f)
	{
		// Do nothing
	}

	__host__ __device__
	ImplicitCollider(ColliderType t, Vector3 c, Vector3 p = Vector3(0, 0, 0), Vector3 v = Vector3(0, 0, 0), float f = 0.1f) :
		type(t), center(c), param(p), velocity(v), coeffFriction(f)
	{
		if (p == Vector3(0, 0, 0))
		{
			if (t == ColliderType::HALF_PLANE)
			{
				param = Vector3(0, 1, 0);
			}
			else if (t == ColliderType::SPHERE)
			{
				param = Vector3(0.5f, 0, 0);
			}
		}
	}

	__host__ __device__
	ImplicitCollider(const ImplicitCollider& collider) :
		type(collider.type), center(collider.center), param(collider.param),
		velocity(collider.velocity), coeffFriction(collider.coeffFriction)
	{
		// Do nothing
	}

	__host__ __device__
	void ApplyTransformation(const glm::mat4 &ctm)
	{
		glm::vec4 c = ctm * glm::vec4(glm::vec3(0, 0, 0), 1.f);
		center = Vector3(c.x, c.y, c.z);
		
		switch (type)
		{
		case ColliderType::HALF_PLANE:
		{
			glm::vec4 n = ctm * glm::vec4(glm::vec3(0, 1, 0), 0.f);
			param = Vector3(n.x, n.y, n.z);
			break;
		}
		case ColliderType::SPHERE:
		{
			const float* m = glm::value_ptr(ctm);
			// Assumes uniform scale
			param.x = sqrtf(m[0] * m[0] + m[1] * m[1] + m[2] * m[2]);
			break;
		}
		}
	}
};

#endif