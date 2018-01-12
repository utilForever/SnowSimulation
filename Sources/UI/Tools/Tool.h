/*************************************************************************
> File Name: Tool.h
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Common tool class of snow simulation.
> Created Time: 2018/01/04
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#ifndef SNOW_SIMULATION_TOOL_H
#define SNOW_SIMULATION_TOOL_H

#ifndef GLM_FORCE_RADIANS
#define GLM_FORCE_RADIANS
#endif

#include <glm/mat4x4.hpp>

class ViewPanel;
struct Vector3;

class Tool
{
public:
	enum class Type
	{
		SELECTION,
		MOVE,
		ROTATE,
		SCALE,
		VELOCITY
	};

	Tool(ViewPanel* panel, Type t);
	virtual ~Tool() = default;

	virtual void MousePressed();
	virtual void MouseMoved();
	virtual void MouseReleased();

	virtual void Update();

	virtual void Render();

	static Vector3 GetAxialColor(unsigned int axis);

protected:
	static glm::mat4 GetAxialBasis(unsigned int axis);

	float GetHandleSize(const Vector3& center) const;

	ViewPanel* m_panel;
	bool m_mouseDown;

	Type m_type;
};

#endif