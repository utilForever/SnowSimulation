/*************************************************************************
> File Name: VelocityTool.cpp
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Velocity tool of snow simulation.
> Created Time: 2018/01/11
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
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
#include <UI/Tools/VelocityTool.h>
#include <Viewport/Camera.h>
#include <Viewport/Viewport.h>

#ifndef GLM_FORCE_RADIANS
#define GLM_FORCE_RADIANS
#endif

#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <QVector>

#include <iostream>

VelocityTool::VelocityTool(ViewPanel *panel, Type t) :
	SelectionTool(panel, t), m_axisSelection(Picker::NO_PICK), m_vecSelection(Picker::NO_PICK),
	m_active(false), m_rotating(false), m_scaling(false), m_center(0, 0, 0),
	m_scale(1.f), m_vbo(0), m_vboSize(0)
{
	// Do nothing
}

VelocityTool::~VelocityTool()
{
	DeleteVBO();
}

void VelocityTool::MousePressed()
{
	if (m_active)
	{
		m_axisSelection = GetAxisPick();
		m_vecSelection = GetVelocityVectorPick();
		m_rotating = (m_axisSelection != Picker::NO_PICK && m_vecSelection == Picker::NO_PICK);
		m_scaling = (m_vecSelection != Picker::NO_PICK);

		if (m_axisSelection == Picker::NO_PICK && m_vecSelection == Picker::NO_PICK)
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

void VelocityTool::MouseMoved()
{
	if (m_rotating)
	{
		float theta0 = IntersectAxis(UserInput::GetMousePos() - UserInput::GetMouseMove());
		float theta1 = IntersectAxis(UserInput::GetMousePos());
		float theta = theta1 - theta0;
		glm::mat4 Tinv = glm::translate(glm::mat4(1.f), glm::vec3(-m_center));
		glm::mat4 T = glm::translate(glm::mat4(1.f), glm::vec3(m_center));
		glm::vec3 axis(0, 0, 0); axis[m_axisSelection] = 1.f;
		glm::mat4 R = glm::rotate(glm::mat4(1.f), theta, axis);
		glm::mat4 transform = T * R * Tinv;

		for (SceneNodeIterator iter = m_panel->m_scene->Begin(); iter.IsValid(); ++iter)
		{
			if ((*iter)->HasRenderable() && (*iter)->GetRenderable()->GetSelected() &&
				(*iter)->GetType() != SceneNode::Type::SIMULATION_GRID)
			{
				glm::mat4 ctm = (*iter)->GetCTM();

				(*iter)->GetRenderable()->RotateVelocity(transform, ctm);
				(*iter)->GetRenderable()->UpdateMeshVelocity();
				m_panel->CheckSelected();
			}
		}
	}
	if (m_scaling)
	{
		const float scaleFactor = 23.0f;
		const glm::ivec2& p0 = UserInput::GetMousePos() - UserInput::GetMouseMove();
		const glm::ivec2& p1 = UserInput::GetMousePos();

		for (SceneNodeIterator iter = m_panel->m_scene->Begin(); iter.IsValid(); ++iter)
		{
			if ((*iter)->HasRenderable() && (*iter)->GetRenderable()->GetSelected() &&
				(*iter)->GetType() != SceneNode::Type::SIMULATION_GRID)
			{
				float t0, t1;
				glm::vec3 velVec = (*iter)->GetRenderable()->GetVelocityVector();
				
				t0 = IntersectVelVec(p0, velVec);
				t1 = IntersectVelVec(p1, velVec);
				(*iter)->GetRenderable()->SetVelocityMagnitude((*iter)->GetRenderable()->GetVelocityMagnitude() + (t1 - t0) * scaleFactor);
				(*iter)->GetRenderable()->UpdateMeshVelocity();
				
				m_panel->CheckSelected();
			}
		}
	}

	Update();
}

void VelocityTool::MouseReleased()
{
	m_axisSelection = Picker::NO_PICK;
	m_vecSelection = Picker::NO_PICK;
	m_scaling = false;
	m_rotating = false;
	
	SelectionTool::MouseReleased();
	Update();
}

void VelocityTool::Update()
{
	if ((m_active = HasRotatableSelection(m_center)))
	{
		m_scale = GetHandleSize(m_center);
	}
}

void VelocityTool::Render()
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
		
		glPopAttrib();
		glPopAttrib();
		glPopAttrib();
	}
}

void VelocityTool::RenderAxis(unsigned int i) const
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
	glDrawArrays(GL_LINE_LOOP, 0, m_vboSize);
	glDisableClientState(GL_VERTEX_ARRAY);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	
	glPopMatrix();
}

unsigned int VelocityTool::GetAxisPick() const
{
	unsigned int pick = Picker::NO_PICK;

	if (m_active)
	{
		m_panel->m_viewport->LoadPickMatrices(UserInput::GetMousePos(), 6.f);
		
		Picker picker(3);
		for (unsigned int i = 0; i < 3; ++i)
		{
			picker.SetObjectIndex(i);
			RenderAxis(i);
		}

		pick = picker.GetPick();

		m_panel->m_viewport->PopMatrices();
	}

	return pick;
}

unsigned int VelocityTool::GetVelocityVectorPick() const
{
	unsigned int pick = Picker::NO_PICK;

	m_panel->m_viewport->LoadPickMatrices(UserInput::GetMousePos(), 3.f);

	QList<SceneNode*> renderables;
	for (SceneNodeIterator iter = m_panel->m_scene->Begin(); iter.IsValid(); ++iter)
	{
		if ((*iter)->HasRenderable() && (*iter)->GetRenderable()->GetSelected())
		{
			renderables += (*iter);
		}
	}

	if (!renderables.empty())
	{
		Picker picker(renderables.size());
		
		for (int i = 0; i < renderables.size(); ++i)
		{
			glMatrixMode(GL_MODELVIEW);
			glPushMatrix();
			glMultMatrixf(glm::value_ptr(renderables[i]->GetCTM()));

			picker.SetObjectIndex(i);
			renderables[i]->GetRenderable()->RenderVelocityForPicker();
			
			glPopMatrix();
		}

		pick = picker.GetPick();
	}

	return pick;
}

float VelocityTool::IntersectVelocityVector(const glm::ivec2& mouse, const glm::vec3& velocityVector) const
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

	float t = (0 - origin[axis]) / direction[axis];
	Vector3 point = origin + t * direction;
	
	return Vector3::Dot(velocityVector, point - m_center);
}

float VelocityTool::IntersectAxis(const glm::ivec2 &mouse) const
{
	glm::vec2 uv = glm::vec2(static_cast<float>(mouse.x) / m_panel->width(), static_cast<float>(mouse.y) / m_panel->height());
	Vector3 direction = m_panel->m_viewport->GetCamera()->GetCameraRay(uv);
	Vector3 origin = m_panel->m_viewport->GetCamera()->GetPosition();
	Vector3 normal(0, 0, 0); normal[m_axisSelection] = 1.f;
	float t = (m_center[m_axisSelection] - origin[m_axisSelection]) / direction[m_axisSelection];
	Vector3 circle = (origin + t * direction) - m_center;
	float y = circle[(m_axisSelection + 2) % 3];
	float x = circle[(m_axisSelection + 1) % 3];

	return atan2(y, x);
}

bool VelocityTool::HasVBO() const
{
	return m_vbo > 0 && glIsBuffer(m_vbo);
}

void VelocityTool::BuildVBO()
{
	DeleteVBO();

	QVector<Vector3> data;

	static const int resolution = 60;
	static const float dTheta = 2.f * M_PI / resolution;

	for (int i = 0; i < resolution; ++i)
	{
		data += Vector3(cosf(i * dTheta), 0.f, -sinf(i * dTheta));
	}

	glGenBuffers(1, &m_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
	glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(Vector3), data.data(), GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	m_vboSize = data.size();
}

void VelocityTool::DeleteVBO()
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