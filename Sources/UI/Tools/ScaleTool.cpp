/*************************************************************************
> File Name: ScaleTool.cpp
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Scale tool of snow simulation.
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
#include <UI/Tools/ScaleTool.h>
#include <Viewport/Camera.h>
#include <Viewport/Viewport.h>

#ifndef GLM_FORCE_RADIANS
#define GLM_FORCE_RADIANS
#endif

#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

constexpr float SCALE = 0.01f;

ScaleTool::ScaleTool(ViewPanel *panel, Type t) :
	SelectionTool(panel, t), m_axisSelection(Picker::NO_PICK),
	m_active(false), m_scaling(false), m_center(0, 0, 0),
	m_scale(1.f), m_mouseDownPos(0, 0), m_transformInverse(1.f),
	m_transform(1.f), m_vbo(0), m_vboSize(0), m_radius(0.05f)
{
	// Do nothing
}

ScaleTool::~ScaleTool()
{
	DeleteVBO();
}

void ScaleTool::MousePressed()
{
	if (m_active)
	{
		m_transform = m_transformInverse = glm::mat4(1.f);
		m_axisSelection = GetAxisPick();
		m_scaling = (m_axisSelection != Picker::NO_PICK);
		
		if (m_axisSelection == Picker::NO_PICK)
		{
			SelectionTool::MousePressed();
		}
		else if (m_axisSelection == 3)
		{
			m_mouseDownPos = UserInput::GetMousePos();
		}
	}
	else
	{
		SelectionTool::MousePressed();
	}

	Update();
}

void ScaleTool::MouseMoved()
{
	if (m_scaling)
	{
		const glm::ivec2& p0 = UserInput::GetMousePos() - UserInput::GetMouseMove();
		const glm::ivec2& p1 = UserInput::GetMousePos();
		float t0, t1;
		glm::mat4 transform = glm::mat4(1.f);
		glm::mat4 uniformScale = glm::mat4(1.f);
		
		if (m_axisSelection < 3)
		{
			t0 = IntersectAxis(p0);
			t1 = IntersectAxis(p1);

			if (fabsf(t1) > 1e-6)
			{
				float t = t1 / t0;
				glm::vec3 scale = glm::vec3(1, 1, 1); scale[m_axisSelection] = t;
				transform = glm::scale(glm::mat4(1.f), scale);
				uniformScale = glm::scale(glm::mat4(1.f), glm::vec3(t));
			}
		}
		else
		{
			float d = 1.f + SCALE * (p1.x - m_mouseDownPos.x);
			
			if (fabsf(d) > 1e-6)
			{
				m_transform = glm::scale(glm::mat4(1.f), glm::vec3(d, d, d));
				transform = m_transform * m_transformInverse;
				uniformScale = transform;
				float* i = glm::value_ptr(m_transform);
				m_transformInverse = glm::mat4(
					1.f / i[0], 0.f, 0.f, 0.f,
					0.f, 1.f / i[5], 0.f, 0.f,
					0.f, 0.f, 1.f / i[10], 0.f,
					0.f, 0.f, 0.f, 1.f);
			}
		}

		glm::mat4 T = glm::translate(glm::mat4(1.f), glm::vec3(m_center));
		glm::mat4 Tinv = glm::translate(glm::mat4(1.f), glm::vec3(-m_center));
		
		transform = T * transform * Tinv;
		uniformScale = T * uniformScale * Tinv;
		
		for (SceneNodeIterator iter = m_panel->m_scene->Begin(); iter.IsValid(); ++iter)
		{
			if ((*iter)->HasRenderable() && (*iter)->GetRenderable()->GetSelected() && (*iter)->GetType() != SceneNode::Type::SIMULATION_GRID)
			{
				if ((*iter)->GetType() == SceneNode::Type::SCENE_COLLIDER)
				{
					(*iter)->ApplyTransformation(uniformScale);
				}
				else
				{
					(*iter)->ApplyTransformation(transform);
				}
			}
		}
	}

	Update();
}

void ScaleTool::MouseReleased()
{
	m_axisSelection = Picker::NO_PICK;
	m_scaling = false;

	SelectionTool::MouseReleased();
	Update();
}

void ScaleTool::Update()
{
	if ((m_active = HasScalableSelection(m_center)))
	{
		m_scale = GetHandleSize(m_center);
	}
}

void ScaleTool::Render()
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

void ScaleTool::RenderAxis(unsigned int i) const
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
	glDrawArrays(GL_QUADS, 2, m_vboSize - 2);
	glDisableClientState(GL_VERTEX_ARRAY);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	
	glPopMatrix();
}

void ScaleTool::RenderCenter() const
{
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();

	glm::mat4 translate = glm::translate(glm::mat4(1.f), glm::vec3(m_center.x, m_center.y - m_scale * (1.f - m_radius), m_center.z));
	glm::mat4 scale = glm::scale(glm::mat4(1.f), glm::vec3(m_scale));
	
	glMultMatrixf(glm::value_ptr(translate * scale));
	glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, sizeof(Vector3), static_cast<void*>(nullptr));
	glDrawArrays(GL_QUADS, 2, m_vboSize - 2);
	glDisableClientState(GL_VERTEX_ARRAY);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	
	glPopMatrix();
}

unsigned int ScaleTool::GetAxisPick() const
{
	unsigned int pick = Picker::NO_PICK;
	
	if (m_active)
	{
		m_panel->m_viewport->LoadPickMatrices(UserInput::GetMousePos(), 1.f);
		
		Picker picker(4);
		for (unsigned int i = 0; i < 3; ++i)
		{
			picker.SetObjectIndex(i);
			RenderAxis(i);
		}
		
		picker.SetObjectIndex(3);
		
		RenderCenter();
		
		pick = picker.GetPick();
		
		m_panel->m_viewport->PopMatrices();
	}

	return pick;
}

float ScaleTool::IntersectAxis(const glm::ivec2& mouse) const
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
	Vector3 a = Vector3(0, 0, 0);
	a[m_axisSelection] = 1.f;
	
	return Vector3::Dot(a, point - m_center);
}

bool ScaleTool::HasVBO() const
{
	return m_vbo > 0 && glIsBuffer(m_vbo);
}

void ScaleTool::BuildVBO()
{
	DeleteVBO();

	QVector<Vector3> data;
	data += Vector3(0, 0, 0);
	data += Vector3(0, 1, 0);

	static const int resolution = 60;
	static const float dAngle = 2.f * M_PI / resolution;

	Vector3 center = Vector3(0, 1 - m_radius, 0);

	for (int i = 0; i < resolution; ++i)
	{
		float theta0 = i * dAngle;
		float theta1 = (i + 1) * dAngle;
		float y0 = cosf(theta0);
		float y1 = cosf(theta1);
		float r0 = sinf(theta0);
		float r1 = sinf(theta1);

		for (int j = 0; j < resolution; ++j)
		{
			float phi0 = j * dAngle;
			float phi1 = (j + 1) * dAngle;
			float x0 = cosf(phi0);
			float x1 = cosf(phi1);
			float z0 = -sinf(phi0);
			float z1 = -sinf(phi1);

			data += (center + m_radius * Vector3(r0 * x0, y0, r0 * z0));
			data += (center + m_radius * Vector3(r1 * x0, y1, r1 * z0));
			data += (center + m_radius * Vector3(r1 * x1, y1, r1 * z1));
			data += (center + m_radius * Vector3(r0 * x1, y0, r0 * z1));
		}
	}

	glGenBuffers(1, &m_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
	glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(Vector3), data.data(), GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	m_vboSize = data.size();
}

void ScaleTool::DeleteVBO()
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