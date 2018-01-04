/*************************************************************************
> File Name: Tool.cpp
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Common tool class of snow simulation.
> Created Time: 2018/01/04
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#include <CUDA/Vector.h>
#include <UI/ViewPanel.h>
#include <UI/Tools/Tool.h>
#include <Viewport/Camera.h>
#include <Viewport/Viewport.h>

constexpr int HANDLE_SIZE = 100;

Tool::Tool(ViewPanel* panel, Type t) :
    m_panel(panel), m_mouseDown(false), m_type(t)
{
    // Do nothing
}

void Tool::MousePressed()
{
    m_mouseDown = true;
}

void Tool::MouseMoved()
{
    // Do nothing
}

void Tool::MouseReleased()
{
    m_mouseDown = false;
}

void Tool::Update()
{
    // Do nothing
}

void Tool::Render()
{
    // Do nothing
}

Vector3 Tool::GetAxialColor(unsigned int axis)
{
    switch (axis)
    {
    case 0:
        return Vector3(186 / 255., 70 / 255., 85 / 255.);
    case 1:
        return Vector3(91 / 255., 180 / 255., 71 / 255.);
    case 2:
        return Vector3(79 / 255., 79 / 255., 190 / 255.);
    default:
        return Vector3(190 / 255., 190 / 255., 69 / 255.);
    }
}

glm::mat4 Tool::GetAxialBasis(unsigned int axis)
{
    const float m[] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
    unsigned int x = (axis + 2) % 3;
    unsigned int y = axis;
    unsigned int z = (axis + 1) % 3;
    
    return glm::mat4(
        m[x], m[3 + x], m[6 + x], 0,
        m[y], m[3 + y], m[6 + y], 0,
        m[z], m[3 + z], m[6 + z], 0,
        0, 0, 0, 1);
}

float Tool::GetHandleSize(const Vector3& center) const
{
    glm::vec3 c(center);
    Camera* camera = m_panel->m_viewport->GetCamera();
    float distance = glm::length(c - camera->GetPosition());
    glm::vec2 uv = camera->GetProjection(c);
    glm::vec3 ray = camera->GetCameraRay(uv + glm::vec2(0.f, HANDLE_SIZE / static_cast<float>(m_panel->height())));
    glm::vec3 point = camera->GetPosition() + distance * ray;

    return glm::length(point - c);
}