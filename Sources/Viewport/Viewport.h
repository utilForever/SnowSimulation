/*************************************************************************
> File Name: Viewport.h
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Viewport of snow simulation.
> Created Time: 2018/01/04
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#ifndef SNOW_SIMULATION_VIEWPORT_H
#define SNOW_SIMULATION_VIEWPORT_H

#ifndef GLM_FORCE_RADIANS
#define GLM_FORCE_RADIANS
#endif

#include <glm/vec3.hpp>

class QWidget;
class Camera;

class Viewport
{
public:
    enum class State
    {
        IDLE,
        PANNING,
        ZOOMING,
        TUMBLING
    };

    Viewport();
    ~Viewport();

    Camera* GetCamera() const;

    void LoadMatrices() const;
    static void PopMatrices();
    void LoadPickMatrices(const glm::ivec2& click, float size) const;

    void Push() const;
    void Pop() const;

    void Orient(const glm::vec3& eye, const glm::vec3& lookAt, const glm::vec3& up);

    void SetDimensions(int width, int height);

    void SetState(State state);
    State GetState() const;

    void MouseMoved();

    void DrawAxis();

private:
    State m_state;
    Camera* m_camera;
    int m_width, m_height;
};

#endif