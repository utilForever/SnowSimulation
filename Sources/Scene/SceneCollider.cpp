/*************************************************************************
> File Name: SceneCollider.cpp
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Scene collider of snow simulation.
> Created Time: 2018/01/07
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#include <Geometry/BBox.h>
#include <IO/ObjParser.h>
#include <Scene/SceneCollider.h>
#include <Simulation/ImplicitCollider.h>

#include <QGL.h>    

SceneCollider::SceneCollider(ImplicitCollider* collider) : m_collider(collider)
{
	InitializeMesh();
	SceneCollider::UpdateMeshVelocity();
}

SceneCollider::~SceneCollider()
{
	if (m_collider != nullptr)
	{
		delete m_collider;
		m_collider = nullptr;
	}
}

void SceneCollider::Render()
{
	m_mesh->Render();
}

void SceneCollider::RenderForPicker()
{
	m_mesh->RenderForPicker();
}

void SceneCollider::RenderVelocityForPicker()
{
	UpdateMeshVelocity();
	m_mesh->RenderVelocityForPicker();
}

void SceneCollider::UpdateMeshVelocity()
{
	m_mesh->SetVelocityMagnitude(m_VelocityMagnitude);
	m_mesh->SetVelocityVector(m_velocityVector);
	m_mesh->UpdateMeshVelocity();
}

float SceneCollider::GetSphereRadius()
{
	return .01f;
}

BBox SceneCollider::GetBBox(const glm::mat4& ctm)
{
	return m_mesh->GetBBox(ctm);
}

Vector3 SceneCollider::GetCentroid(const glm::mat4& ctm)
{
	return m_mesh->GetCentroid(ctm);
}

void SceneCollider::InitializeMesh()
{
	QList<Mesh*> colliderMeshes;

	switch (m_collider->type)
	{
	case ColliderType::SPHERE:
		ObjParser::Load("Datas/Models/SphereCol.obj", colliderMeshes);
		break;
	case ColliderType::HALF_PLANE:
		ObjParser::Load("Datas/Models/Plane.obj", colliderMeshes);
		break;
	default:
		break;
	}

	m_mesh = colliderMeshes[0];
	m_mesh->SetType(Mesh::Type::COLLIDER);
}

void SceneCollider::SetSelected(bool selected)
{
	m_selected = selected;
	m_mesh->SetSelected(selected);
}

void SceneCollider::SetCTM(const glm::mat4& ctm)
{
	m_mesh->SetCTM(ctm);
}

void SceneCollider::RenderVelocity(bool velTool)
{
	m_mesh->RenderVelocity(velTool);
}

ImplicitCollider* SceneCollider::GetImplicitCollider()
{
	return m_collider;
}