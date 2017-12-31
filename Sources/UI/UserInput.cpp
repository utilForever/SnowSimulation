/*************************************************************************
> File Name: UserInput.cpp
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: User input UI of snow simulation.
> Created Time: 2017/12/31
> Copyright (c) 2017, Chan-Ho Chris Ohk
*************************************************************************/
#include <UI/UserInput.h>

UserInput* UserInput::m_instance = nullptr;

UserInput::UserInput() :
	m_mousePos(glm::ivec2(0, 0)), m_mouseMove(glm::ivec2(0, 0)),
	m_button(Qt::NoButton), m_modifiers(Qt::NoModifier)
{
	// Do nothing
}

UserInput* UserInput::GetInstance()
{
	if (m_instance == nullptr)
	{
		m_instance = new UserInput;
	}

	return m_instance;
}

void UserInput::DeleteInstance()
{
	if (m_instance != nullptr)
	{
		delete m_instance;
		m_instance = nullptr;
	}
}

void UserInput::Update(QMouseEvent* event)
{
	GetInstance()->m_mouseMove = glm::ivec2(
		event->pos().x() - GetInstance()->m_mousePos.x,
		event->pos().y() - GetInstance()->m_mousePos.y);
	GetInstance()->m_mousePos = glm::ivec2(event->pos().x(), event->pos().y());
	GetInstance()->m_button = event->button();
	GetInstance()->m_modifiers = event->modifiers();
}

glm::ivec2 UserInput::GetMousePos()
{
	return GetInstance()->m_mousePos;
}

glm::ivec2 UserInput::GetMouseMove()
{
	return GetInstance()->m_mouseMove;
}

bool UserInput::IsMouseLeft()
{
	return GetInstance()->m_button == Qt::LeftButton;
}

bool UserInput::IsMouseRight()
{
	return GetInstance()->m_button == Qt::RightButton;
}

bool UserInput::IsMouseMiddle()
{
	return GetInstance()->m_button == Qt::MiddleButton;
}

bool UserInput::IsAltKey()
{
	return GetInstance()->m_modifiers == Qt::AltModifier;
}

bool UserInput::IsCtrlKey()
{
	return GetInstance()->m_modifiers == Qt::ControlModifier;
}

bool UserInput::IsShiftKey()
{
	return GetInstance()->m_modifiers == Qt::ShiftModifier;
}