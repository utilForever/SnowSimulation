/*************************************************************************
> File Name: VelocityTool.h
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Velocity tool of snow simulation.
> Created Time: 2018/01/11
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#ifndef SNOW_SIMULATION_VELOCITY_TOOL_H
#define SNOW_SIMULATION_VELOCITY_TOOL_H

#include <CUDA/Vector.h>
#include <UI/Tools/SelectionTool.h>

#ifndef GLM_FORCE_RADIANS
#define GLM_FORCE_RADIANS
#endif

#include <glm/vec2.hpp>

using GLuint = unsigned int;

class VelocityTool : public SelectionTool
{
public:
	VelocityTool(ViewPanel* panel, Type t);
	virtual ~VelocityTool();

	void MousePressed() override;
	void MouseMoved() override;
	void MouseReleased() override;

	void Update() override;

	void Render() override;

protected:
	unsigned int m_axisSelection, m_vecSelection;

	bool m_active;
	bool m_rotating, m_scaling;
	Vector3 m_center;
	float m_scale;

	GLuint m_vbo;
	int m_vboSize;

	void RenderAxis(unsigned int i) const;
	unsigned int GetAxisPick() const;
	unsigned int GetVelocityVectorPick() const;

	float IntersectVelocityVector(const glm::ivec2& mouse, const glm::vec3& velocityVector) const;
	float IntersectAxis(const glm::ivec2& mouse) const;

	bool HasVBO() const;
	void BuildVBO();
	void DeleteVBO();
};

#endif