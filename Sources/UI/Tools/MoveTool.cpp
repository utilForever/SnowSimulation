/*************************************************************************
> File Name: MoveTool.cpp
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Move tool of snow simulation.
> Created Time: 2018/01/11
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#include <Windows.h>

#include <GL/glew.h>
#include <GL/gl.h>

#define _USE_MATH_DEFINES
#include <cmath>

#include <Scene/Scene.h>
#include <Scene/SceneNode.h>
#include <Scene/SceneNodeIterator.h>
#include <UI/Picker.h>
#include <UI/UISettings.h>
#include <UI/UserInput.h>
#include <UI/ViewPanel.h>
#include <UI/Tools/MoveTool.h>
#include <Viewport/Camera.h>
#include <Viewport/Viewport.h>

#ifndef GLM_FORCE_RADIANS
#define GLM_FORCE_RADIANS
#endif

#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

MoveTool::MoveTool(ViewPanel* panel, Type t) :
	SelectionTool(panel, t), m_axisSelection(Picker::NO_PICK),
	m_active(false), m_moving(false), m_center(0, 0, 0),
	m_scale(1.f), m_vbo(0), m_vboSize(0)
{
	// Do nothing
}

MoveTool::~MoveTool()
{
	DeleteVBO();
}

void MoveTool::MousePressed()
{
	if (m_active)
	{
		m_axisSelection = GetAxisPick();
		m_moving = (m_axisSelection != Picker::NO_PICK);
		
		if (m_axisSelection == Picker::NO_PICK)
		{
			SelectionTool::MousePressed();
		}
	}
	else
	{
		SelectionTool::MousePressed();
	}

	Update();
}

void MoveTool::MouseMoved()
{
	if (m_moving)
	{
		const glm::ivec2& p0 = UserInput::GetMousePos() - UserInput::GetMouseMove();
		const glm::ivec2& p1 = UserInput::GetMousePos();
		glm::mat4 transform;
		
		if (m_axisSelection < 3)
		{
			float t0 = IntersectAxis(p0);
			float t1 = IntersectAxis(p1);
			float t = t1 - t0;

			glm::vec3 translate = glm::vec3(0, 0, 0);
			translate[m_axisSelection] = t;
			
			transform = glm::translate(glm::mat4(1.f), translate);
		}
		else
		{
			Camera* camera = m_panel->m_viewport->GetCamera();
			float depth = Vector3::Dot((m_center - Vector3(camera->GetPosition())), camera->GetLook());
			Vector3 ray0 = camera->GetCameraRay(glm::vec2(p0.x / (float)m_panel->width(), p0.y / (float)m_panel->height()));
			float t0 = depth / Vector3::Dot(ray0, camera->GetLook());
			Vector3 ray1 = camera->GetCameraRay(glm::vec2(p1.x / (float)m_panel->width(), p1.y / (float)m_panel->height()));
			float t1 = depth / Vector3::Dot(ray1, camera->GetLook());
			glm::vec3 translate = t1 * ray1 - t0 * ray0;
			
			transform = glm::translate(glm::mat4(1.f), translate);
		}

		for (SceneNodeIterator iter = m_panel->m_scene->Begin(); iter.IsValid(); ++iter)
		{
			if ((*iter)->HasRenderable() && (*iter)->GetRenderable()->GetSelected())
			{
				(*iter)->ApplyTransformation(transform);
			}
		}
	}

	Update();
}

void MoveTool::MouseReleased()
{
	m_axisSelection = Picker::NO_PICK;
	m_moving = false;
	
	SelectionTool::MouseReleased();
	Update();
}

void MoveTool::Update()
{
	if ((m_active = HasSelection(m_center)))
	{
		m_scale = GetHandleSize(m_center);
	}
}

void MoveTool::Render()
{
	if (m_active)
	{
		if (!HasVBO())
		{
			BuildVBO();
		}

		glPushAttrib(GL_DEPTH_BUFFER_BIT);
		glDisable(GL_DEPTH_TEST);
		glPushAttrib(GL_LIGHTING_BIT);
		glDisable(GL_LIGHTING);
		glPushAttrib(GL_COLOR_BUFFER_BIT);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glEnable(GL_LINE_SMOOTH);
		glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

		for (unsigned int i = 0; i < 3; ++i)
		{
			glColor3fv(GetAxialColor((i == m_axisSelection) ? 3 : i).data);
			RenderAxis(i);
		}

		glColor3fv(GetAxialColor(3).data);
		
		RenderCenter();
		
		glPopAttrib();
		glPopAttrib();
		glPopAttrib();
	}
}

void MoveTool::RenderAxis(unsigned int i) const
{
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();

	glm::mat4 translate = glm::translate(glm::mat4(1.f), glm::vec3(m_center));
	glm::mat4 basis = glm::scale(GetAxialBasis(i), glm::vec3(m_scale));
	
	glMultMatrixf(glm::value_ptr(translate * basis));
	glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, sizeof(Vector3), static_cast<void*>(nullptr));
	glLineWidth(2.f);
	glDrawArrays(GL_LINES, 0, 2);
	glDrawArrays(GL_TRIANGLES, 2, m_vboSize - (2 + 24));
	glDisableClientState(GL_VERTEX_ARRAY);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	
	glPopMatrix();
}

void MoveTool::RenderCenter() const
{
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();

	glm::mat4 translate = glm::translate(glm::mat4(1.f), glm::vec3(m_center));
	glm::mat4 scale = glm::scale(glm::mat4(1.f), glm::vec3(m_scale));

	glMultMatrixf(glm::value_ptr(translate * scale));
	glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, sizeof(Vector3), static_cast<void*>(nullptr));
	glDrawArrays(GL_QUADS, m_vboSize - 24, 24);
	glDisableClientState(GL_VERTEX_ARRAY);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glPopMatrix();
}

unsigned int MoveTool::GetAxisPick() const
{
	unsigned int pick = Picker::NO_PICK;
	
	if (m_active)
	{
		m_panel->m_viewport->LoadPickMatrices(UserInput::GetMousePos(), 6.f);
		
		Picker picker(4);
		for (unsigned int i = 0; i < 3; ++i)
		{
			picker.SetObjectIndex(i);
			RenderAxis(i);
		}

		picker.SetObjectIndex(4);
		
		RenderCenter();
		
		pick = picker.GetPick();

		Viewport::PopMatrices();
	}

	return pick;
}

float MoveTool::IntersectAxis(const glm::ivec2& mouse) const
{
	glm::vec2 uv = glm::vec2(static_cast<float>(mouse.x) / m_panel->width(), static_cast<float>(mouse.y) / m_panel->height());
	Vector3 direction = m_panel->m_viewport->GetCamera()->GetCameraRay(uv);
	Vector3 origin = m_panel->m_viewport->GetCamera()->GetPosition();
	unsigned int majorAxis = direction.GetMajorAxis();
	int axis = majorAxis;
	
	if (majorAxis == m_axisSelection)
	{
		axis = (majorAxis == 0) ? 1 : 0;
	}

	float t = (m_center[axis] - origin[axis]) / direction[axis];
	Vector3 point = origin + t * direction;
	Vector3 a = Vector3(0, 0, 0); a[m_axisSelection] = 1.f;
	
	return Vector3::Dot(a, point - m_center);
}

bool MoveTool::HasVBO() const
{
	return m_vbo > 0 && glIsBuffer(m_vbo);
}

void MoveTool::BuildVBO()
{
	DeleteVBO();

	QVector<Vector3> data;

	// Axis
	data += Vector3(0, 0, 0);
	data += Vector3(0, 1, 0);

	// Cone
	static const int resolution = 60;
	static const float dTheta = 2.f * M_PI / resolution;
	static const float coneHeight = 0.1f;
	static const float coneRadius = 0.05f;
	for (int i = 0; i < resolution; ++i)
	{
		data += Vector3(0, 1, 0);
		float theta0 = i * dTheta;
		float theta1 = (i + 1)*dTheta;
		data += (Vector3(0, 1 - coneHeight, 0) + coneRadius * Vector3(cosf(theta0), 0, -sinf(theta0)));
		data += (Vector3(0, 1 - coneHeight, 0) + coneRadius * Vector3(cosf(theta1), 0, -sinf(theta1)));
	}

	// Cube
	static const float s = 0.05f;
	data += Vector3(-s, s, -s);
	data += Vector3(-s, -s, -s);
	data += Vector3(-s, -s, s);
	data += Vector3(-s, s, s);
	data += Vector3(s, s, s);
	data += Vector3(s, -s, s);
	data += Vector3(s, -s, -s);
	data += Vector3(s, s, -s);
	data += Vector3(-s, s, s);
	data += Vector3(-s, -s, s);
	data += Vector3(s, -s, s);
	data += Vector3(s, s, s);
	data += Vector3(s, s, -s);
	data += Vector3(s, -s, -s);
	data += Vector3(-s, -s, -s);
	data += Vector3(-s, s, -s);
	data += Vector3(-s, s, -s);
	data += Vector3(-s, s, s);
	data += Vector3(s, s, s);
	data += Vector3(s, s, -s);
	data += Vector3(s, s, -s);
	data += Vector3(s, s, s);
	data += Vector3(-s, s, s);
	data += Vector3(-s, s, -s);

	glGenBuffers(1, &m_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
	glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(Vector3), data.data(), GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	m_vboSize = data.size();
}

void MoveTool::DeleteVBO()
{
	if (m_vbo > 0)
	{
		glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
		if (glIsBuffer(m_vbo))
		{
			glDeleteBuffers(1, &m_vbo);
		}
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		
		m_vbo = 0;
	}
}