/*************************************************************************
> File Name: Renderable.h
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: The common renderable header file of snow simulation.
> Created Time: 2017/08/01
> Copyright (c) 2017, Chan-Ho Chris Ohk
*************************************************************************/
#ifndef SNOW_SIMULATION_RENDERABLE_H
#define SNOW_SIMULATION_RENDERABLE_H

#ifndef GLM_FORCE_RADIANS
#define GLM_FORCE_RADIANS
#endif

#include "glm/mat4x4.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/geometric.hpp"
#include "CUDA/vector.h"

// Forward declaration
struct BBox;

class Renderable
{
public:
	Renderable()
	{
		// Do nothing
	}

	virtual ~Renderable()
	{
		// Do nothing
	}

	virtual void Render()
	{
		// Do nothing
	}

	// Skip fancy rendering and just put the primitives
	// onto the frame buffer for pick testing.
	virtual void RenderForPicker()
	{
		// Do nothing
	}

	virtual void RenderVelocityForPicker()
	{
		// Do nothing
	}

	// These functions are used by the SceneNodes to cache their renderable's
	// bounding box or centroid. The object computes its bounding box or
	// centroid in its local coordinate frame, and then transforms it to
	// the SceneNode's using the CTM;
	virtual BBox GetBBox(const glm::mat4& ctm = glm::mat4(1.f)) = 0;

	virtual vec3 GetCentroid(const glm::mat4& ctm = glm::mat4(1.f)) = 0;

	// Used for scene interaction.
	virtual void SetSelected(bool selected)
	{
		m_selected = selected;
	}

	bool GetSelected() const
	{
		return m_selected;
	}

	virtual void RotateVelocity(const glm::mat4& transform, const glm::mat4& ctm)
	{
		glm::vec4 v = glm::vec4(GetWorldVelocity(ctm), 0);

		if (fabs(glm::length(v) - 0) < std::numeric_limits<float>::epsilon())
		{
			return;
		}

		v = transform * v;
		v = glm::normalize(glm::inverse(ctm) * v);
		m_velocity = glm::vec3(v.x, v.y, v.z);
	}

	virtual void SetMagnitude(const float m)
	{
		m_magnitude = m;
	}

	virtual void SetVelocity(const glm::vec3 &vec)
	{
		m_velocity = vec;
	}

	virtual void UpdateMeshVelocity()
	{
		// Do nothing
	}

	virtual void RenderVelocity(bool velTool)
	{
		// Do nothing
	}

	virtual float GetMagnitude()
	{
		return m_magnitude;
	}

	virtual glm::vec3 GetVelocity()
	{
		return m_velocity;
	}

	virtual void SetCTM(const glm::mat4 &ctm)
	{
		m_ctm = ctm;
	}

	virtual glm::vec3 GetWorldVelocity(const glm::mat4& ctm)
	{
		if (fabs(m_magnitude - 0) < std::numeric_limits<float>::epsilon())
		{
			return glm::vec3(0);
		}

		glm::vec3 v = glm::vec3(m_velocity);
		glm::vec4 vWorld = ctm * glm::vec4(v, 0);
		
		return glm::normalize(glm::vec3(vWorld.x, vWorld.y, vWorld.z));
	}

protected:
	bool m_selected = false;
	glm::vec3 m_velocity;
	float m_magnitude = 0.0f;
	glm::mat4 m_ctm;
};

#endif