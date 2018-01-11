/*************************************************************************
> File Name: MoveTool.h
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Move tool of snow simulation.
> Created Time: 2018/01/11
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#ifndef SNOW_SIMULATION_MOVE_TOOL_H
#define SNOW_SIMULATION_MOVE_TOOL_H

#include <CUDA/Vector.h>
#include <UI/Tools/SelectionTool.h>

#ifndef GLM_FORCE_RADIANS
#define GLM_FORCE_RADIANS
#endif

#include <glm/vec2.hpp>

using GLuint = unsigned int;

class MoveTool : public SelectionTool
{
public:
    MoveTool(ViewPanel* panel, Type t);
    virtual ~MoveTool();

    void MousePressed() override;
    void MouseMoved() override;
    void MouseReleased() override;

    void Update() override;

    void Render() override;

protected:
    void RenderAxis(unsigned int i) const;
    void RenderCenter() const;
    unsigned int GetAxisPick() const;
    float IntersectAxis(const glm::ivec2& mouse) const;

    bool HasVBO() const;
    void BuildVBO();
    void DeleteVBO();

    unsigned int m_axisSelection;

    bool m_active;
    bool m_moving;
    Vector3 m_center;
    float m_scale;

    GLuint m_vbo;
    int m_vboSize;
};

#endif