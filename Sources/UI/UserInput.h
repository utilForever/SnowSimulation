/*************************************************************************
> File Name: UserInput.h
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: User input UI of snow simulation.
> Created Time: 2017/12/31
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#ifndef SNOW_SIMULATION_USER_INPUT_H
#define SNOW_SIMULATION_USER_INPUT_H

#include <QMouseEvent>

#ifndef GLM_FORCE_RADIANS
#define GLM_FORCE_RADIANS
#endif

#include <glm/vec2.hpp>

class UserInput
{
public:
    static UserInput* GetInstance();
    static void DeleteInstance();

    static void Update(QMouseEvent* event);

    static glm::ivec2 GetMousePos();
    static glm::ivec2 GetMouseMove();

    static bool IsMouseLeft();
    static bool IsMouseRight();
    static bool IsMouseMiddle();

    static bool IsAltKey();
    static bool IsCtrlKey();
    static bool IsShiftKey();

private:
    UserInput();

    glm::ivec2 m_mousePos;
    glm::ivec2 m_mouseMove;
    Qt::MouseButton m_button;
    Qt::KeyboardModifiers m_modifiers;

    static UserInput* m_instance;
};

#endif