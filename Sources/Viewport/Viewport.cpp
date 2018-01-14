/*************************************************************************
> File Name: Viewport.cpp
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Viewport of snow simulation.
> Created Time: 2018/01/05
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#define _USE_MATH_DEFINES
#include <cmath>

#include <CUDA/Vector.h>
#include <UI/UserInput.h>
#include <UI/Tools/Tool.h>
#include <Viewport/Camera.h>
#include <Viewport/Viewport.h>

#include <Windows.h>

#include <GL/glew.h>
#include <GL/gl.h>

#ifndef GLM_FORCE_RADIANS
#define GLM_FORCE_RADIANS
#endif

#include <glm/gtc/type_ptr.hpp>

constexpr float ZOOM_SCALE = 3.f;

Viewport::Viewport() : m_state(State::IDLE), m_width(1000), m_height(1000)
{
	m_camera = new Camera;
	m_camera->SetClip(0.01f, 1000.f);
	m_camera->SetHeightAngle(M_PI / 6.f);
}

Viewport::~Viewport()
{
	if (m_camera != nullptr)
	{
		delete m_camera;
		m_camera = nullptr;
	}
}

Camera* Viewport::GetCamera() const
{
	return m_camera;
}

void Viewport::LoadMatrices() const
{
	glm::mat4 modelview = m_camera->GetModelViewMatrix();
	glm::mat4 projection = m_camera->GetProjectionMatrix();

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadMatrixf(glm::value_ptr(modelview));
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadMatrixf(glm::value_ptr(projection));
}

void Viewport::PopMatrices()
{
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
}

void Viewport::LoadPickMatrices(const glm::ivec2& click, float size) const
{
	const glm::mat4& modelView = m_camera->GetModelViewMatrix();
	const glm::mat4& projection = m_camera->GetProjectionMatrix();

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();

	float width = static_cast<float>(m_width);
	float height = static_cast<float>(m_height);
	float tX = 2.f * (width / 2.f - click.x) / size;
	float tY = 2.f * (click.y - height / 2.f + 1.f) / size;
	
	const glm::mat4 translate = glm::translate(glm::mat4(1.f), glm::vec3(tX, tY, 0.f));
	glMultMatrixf(glm::value_ptr(translate));

	const glm::mat4 scale = glm::scale(glm::mat4(1.f), glm::vec3(width / size, height / size, 1.f));
	glMultMatrixf(glm::value_ptr(scale));
	glMultMatrixf(glm::value_ptr(projection));

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadMatrixf(glm::value_ptr(modelView));
}

void Viewport::Push() const
{
	glViewport(0, 0, m_width, m_height);
	LoadMatrices();
}

void Viewport::Pop() const
{
	PopMatrices();
}

void Viewport::Orient(const glm::vec3& eye, const glm::vec3& lookAt, const glm::vec3& up)
{
	m_camera->Orient(eye, lookAt, up);
}

void Viewport::SetDimensions(int width, int height)
{
	m_camera->SetAspect(float(width) / float(height));
	m_width = width;
	m_height = height;
}

void Viewport::SetState(State state)
{
	m_state = state;
}

Viewport::State Viewport::GetState() const
{
	return m_state;
}

void Viewport::MouseMoved()
{
	glm::ivec2 pos = UserInput::GetMousePos();
	glm::vec2 posf = glm::vec2(pos.x / float(m_width), pos.y / float(m_height));
	glm::ivec2 move = UserInput::GetMouseMove();
	glm::vec2 movef = glm::vec2(move.x / float(m_width), move.y / float(m_height));

	switch (m_state)
	{
	case State::IDLE:
		break;
	case State::PANNING:
	{
		float tanH = tanf(m_camera->GetHeightAngle() / 2.f);
		float tanW = m_camera->GetAspect() * tanH;
		float du = -2.f * movef.x * m_camera->GetFocusDistance() * tanW;
		float dv = 2.f * movef.y * m_camera->GetFocusDistance() * tanH;
		glm::vec3 trans = du * m_camera->GetU() + dv * m_camera->GetV();

		m_camera->Orient(m_camera->GetPosition() + trans, m_camera->GetLookAt() + trans, m_camera->GetUp());
		break;
	}
	case State::ZOOMING:
	{
		float focus = m_camera->GetFocusDistance();
		if (movef.x < 0 && fabsf(movef.x) > focus)
		{
			break;
		}

		glm::vec3 trans = focus * ZOOM_SCALE * movef.x * m_camera->GetW();
		m_camera->Orient(m_camera->GetPosition() + trans, m_camera->GetLookAt(), m_camera->GetUp());
		break;
	}
	case State::TUMBLING:
	{
		float ax = movef.x * 1.5f * M_PI;
		float ay = movef.y * 1.5f * M_PI;
		float alpha = (posf.x - movef.x / 2.f - 0.5f) / 0.5f;
		float yaw = (1.f - fabsf(alpha)) * ay;
		float pitch = ax;
		float roll = alpha * ay;

		glm::vec4 eye = glm::vec4(m_camera->GetPosition(), 1.f);
		glm::vec4 lookAt = glm::vec4(m_camera->GetLookAt(), 1.f);
		glm::mat4 T = glm::translate(glm::mat4(1.f), -glm::vec3(eye.x, eye.y, eye.z));
		glm::mat4 Tinv = glm::translate(glm::mat4(1.f), glm::vec3(eye.x, eye.y, eye.z));
		glm::mat4 RV = glm::rotate(glm::mat4(1.f), -pitch, m_camera->GetV());
		glm::mat4 RU = glm::rotate(glm::mat4(1.f), -yaw, m_camera->GetU());
		glm::mat4 RW = glm::rotate(glm::mat4(1.f), roll, m_camera->GetW());
		
		eye = lookAt + (Tinv * RW * RU * RV * T) * (eye - lookAt);

		glm::vec4 up = RW * RU * RV * glm::vec4(m_camera->GetUp(), 0.f);
		m_camera->Orient(glm::vec3(eye), glm::vec3(lookAt), glm::vec3(up));
		break;
	}
	}
}

void Viewport::DrawAxis()
{
	static constexpr float corner = 50.f;
	static constexpr float distance = 10.f;
	static constexpr float length = 0.25f;

	glm::vec2 uv = glm::vec2(corner / m_width, 1.f - (corner / m_height));
	glm::vec3 c = m_camera->GetPosition() + distance * m_camera->GetCameraRay(uv);
	
	glm::vec3 x = c + length * glm::vec3(1, 0, 0);
	glm::vec3 y = c + length * glm::vec3(0, 1, 0);
	glm::vec3 z = c + length * glm::vec3(0, 0, 1);

	glPushAttrib(GL_DEPTH_BUFFER_BIT);
	glDisable(GL_DEPTH_TEST);
	glPushAttrib(GL_COLOR_BUFFER_BIT);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_LINE_SMOOTH);
	glHint(GL_LINE_SMOOTH, GL_NICEST);
	glLineWidth(1.5f);

	glBegin(GL_LINES);
	{
		glColor3fv(Tool::GetAxialColor(0).data);
		glVertex3f(c.x, c.y, c.z);
		glVertex3f(x.x, x.y, x.z);
		glColor3fv(Tool::GetAxialColor(1).data);
		glVertex3f(c.x, c.y, c.z);
		glVertex3f(y.x, y.y, y.z);
		glColor3fv(Tool::GetAxialColor(2).data);
		glVertex3f(c.x, c.y, c.z);
		glVertex3f(z.x, z.y, z.z);
	}
	glEnd();
	
	glPopAttrib();
	glPopAttrib();
}