/*************************************************************************
> File Name: Camera.h
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Camera of snow simulation.
> Created Time: 2018/01/04
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#ifndef SNOW_SIMULATION_CAMERA_H
#define SNOW_SIMULATION_CAMERA_H

#ifndef GLM_FORCE_RADIANS
#define GLM_FORCE_RADIANS
#endif

#define _USE_MATH_DEFINES
#include <math.h>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>

class Camera
{
public:
	Camera() : m_aspect(1.f), m_near(0.01f), m_far(1e6), m_heightAngle(M_PI / 3.f)
	{
		UpdateProjectionMatrix();
		Orient(glm::vec3(1, 1, 1), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
	}

	void Orient(const glm::vec3& eye, const glm::vec3& lookAt, const glm::vec3& up)
	{
		m_eye = eye;
		m_lookAt = lookAt;
		m_look = glm::normalize(m_lookAt - m_eye);
		m_up = up;
		m_w = -m_look;
		m_v = glm::normalize(m_up - (glm::dot(m_up, m_w)*m_w));
		m_u = glm::cross(m_v, m_w);

		UpdateModelViewMatrix();
	}

	glm::mat4 GetModelViewMatrix() const
	{
		return m_modelview;
	}

	glm::mat4 GetProjectionMatrix() const
	{
		return m_projection;
	}

	glm::vec3 GetPosition() const
	{
		return m_eye;
	}

	glm::vec3 GetLookAt() const
	{
		return m_lookAt;
	}

	glm::vec3 GetLook() const
	{
		return m_look;
	}

	glm::vec3 GetUp() const
	{
		return m_up;
	}

	glm::vec3 GetU() const
	{
		return m_u;
	}

	glm::vec3 GetV() const
	{
		return m_v;
	}

	glm::vec3 GetW() const
	{
		return m_w;
	}

	float GetAspect() const
	{
		return m_aspect;
	}

	void SetAspect(float aspect)
	{
		m_aspect = aspect;
		UpdateProjectionMatrix();
	}

	float GetNear() const
	{
		return m_near;
	}

	float GetFar() const
	{
		return m_far;
	}

	void SetClip(float _near, float _far)
	{
		m_near = _near;
		m_far = _far;
		UpdateProjectionMatrix();
	}

	float GetHeightAngle() const
	{
		return m_heightAngle;
	}

	void SetHeightAngle(float radians)
	{
		m_heightAngle = radians;
		UpdateProjectionMatrix();
	}

	float GetFocusDistance() const
	{
		return glm::length(m_lookAt - m_eye);
	}

	// Returns world space camera ray through [u,v] in [0,1] x [0,1]
	glm::vec3 GetCameraRay(const glm::vec2& uv) const
	{
		glm::vec3 camDir = glm::vec3(2.f * uv.x - 1.f, 1.f - 2.f * uv.y, -1.f / tanf(m_heightAngle / 2.f));
		glm::vec3 worldDir = m_aspect * camDir.x * m_u + camDir.y * m_v + camDir.z * m_w;

		return glm::normalize(worldDir);
	}

	// Returns [u,v] in [0,1] x [0,1] on image plane
	glm::vec2 GetProjection(const glm::vec3& point) const
	{
		glm::vec4 film = m_projection * m_modelview * glm::vec4(point, 1.f);
		film /= film.w;

		return glm::vec2((film.x + 1.f) / 2.f, (1.f - film.y) / 2.f);
	}

private:
	glm::mat4 m_modelview, m_projection;
	glm::vec3 m_eye, m_lookAt;
	glm::vec3 m_look, m_up, m_u, m_v, m_w;
	float m_aspect, m_near, m_far;
	float m_heightAngle;

	void UpdateModelViewMatrix()
	{
		glm::mat4 translation = glm::translate(glm::mat4(1.f), -m_eye);
		glm::mat4 rotation = glm::mat4(
			m_u.x, m_v.x, m_w.x, 0,
			m_u.y, m_v.y, m_w.y, 0,
			m_u.z, m_v.z, m_w.z, 0,
			0, 0, 0, 1);
		m_modelview = rotation * translation;
	}

	void UpdateProjectionMatrix()
	{
		float tanH = tanf(m_heightAngle / 2.f);
		float tanW = m_aspect * tanH;
		glm::mat4 normalizing = glm::scale(glm::mat4(1.f), glm::vec3(1.f / (m_far*tanW), 1.f / (m_far*tanH), 1.f / m_far));
		float c = -m_near / m_far;
		glm::mat4 unhinging = glm::mat4(
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, -1 / (c + 1), -1,
			0, 0, c / (c + 1), 0);
		m_projection = unhinging * normalizing;
	}

};

#endif