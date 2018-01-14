/*************************************************************************
> File Name: Scene.cpp
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Scene of snow simulation.
> Created Time: 2018/01/07
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#include <Geometry/Mesh.h>
#include <IO/ObjParser.h>
#include <Scene/Scene.h>
#include <Scene/SceneCollider.h>
#include <Scene/SceneGrid.h>
#include <Scene/SceneNode.h>
#include <Scene/SceneNodeIterator.h>
#include <Simulation/ImplicitCollider.h>
#include <UI/UISettings.h>

#include <Windows.h>

#include <GL/glew.h>
#include <GL/gl.h>

#ifndef GLM_FORCE_RADIANS
#define GLM_FORCE_RADIANS
#endif

#include <glm/vec4.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/rotate_vector.hpp>

#include <QQueue>

Scene::Scene() : m_root(new SceneNode)
{
	InitSceneGrid();
}

Scene::~Scene()
{
	for (SceneNodeIterator it = Begin(); it.IsValid(); ++it)
	{
		if ((*it)->HasRenderable() == true && (*it)->GetType() == SceneNode::Type::SIMULATION_GRID)
		{
			glm::vec4 point = (*it)->GetCTM() * glm::vec4(0, 0, 0, 1);
			UISettings::gridPosition() = Vector3(point.x, point.y, point.z);
			break;
		}
	}

	UISettings::SaveSettings();
	
	if (m_root != nullptr)
	{
		delete m_root;
		m_root = nullptr;
	}
}

void Scene::Render()
{
	SetupLights();

	// Render opaque objects, then overlay with transparent objects
	m_root->RenderOpaque();
	m_root->RenderTransparent();
}

void Scene::RenderVelocity(bool velTool)
{
	m_root->RenderVelocity(velTool);
}

SceneNode* Scene::GetRoot()
{
	return m_root;
}

SceneNode* Scene::GetSceneGridNode()
{
	for (int i = 0; i < m_root->GetChild().size(); ++i)
	{
		SceneNode* child = m_root->GetChild()[i];

		if (child->HasRenderable() == true && (child->GetType() == SceneNode::Type::SIMULATION_GRID))
		{
			return child;
		}
	}

	return nullptr;
}

SceneNodeIterator Scene::Begin() const
{
	QList<SceneNode*> nodes;
	nodes += m_root;
	int i = 0;

	while (i < nodes.size())
	{
		nodes += nodes[i]->GetChild();
		i++;
	}

	return SceneNodeIterator(nodes);
}

void Scene::DeleteSelectedNodes()
{
	QQueue<SceneNode*> nodes;
	nodes += m_root;

	while (!nodes.empty())
	{
		SceneNode* node = nodes.dequeue();
		if (node->HasRenderable() == true &&
			node->GetType() != SceneNode::Type::SIMULATION_GRID &&
			node->GetRenderable()->GetSelected() == true)
		{
			// Delete node through its parent so that the scene graph is appropriately
			// rid of the deleted node.
			node->GetParent()->DeleteChild(node);
		}
		else
		{
			nodes += node->GetChild();
		}
	}
}

void Scene::LoadMesh(const QString& fileName, glm::mat4 ctm)
{
	QList<Mesh*> meshes;
	ObjParser::Load(fileName, meshes);
	
	for (int i = 0; i < meshes.size(); ++i)
	{
		Mesh* mesh = meshes[i];
		mesh->SetType(Mesh::Type::SNOW_CONTAINER);
		
		SceneNode* node = new SceneNode(SceneNode::Type::SNOW_CONTAINER);
		node->SetRenderable(mesh);
		node->ApplyTransformation(ctm);
		m_root->AddChild(node);
	}
}

void Scene::Reset()
{
	if (m_root != nullptr)
	{
		delete m_root;
		m_root = nullptr;
	}

	m_root = new SceneNode;
}

void Scene::InitSceneGrid()
{
	// Add scene grid
	SceneNode* gridNode = new SceneNode(SceneNode::Type::SIMULATION_GRID);
	glm::mat4 transform = glm::translate(glm::mat4(1.f), glm::vec3(UISettings::gridPosition()));
	gridNode->ApplyTransformation(transform);

	Grid grid;
	grid.pos = Vector3(0, 0, 0);
	grid.dim = UISettings::gridDimensions();
	grid.h = UISettings::gridResolution();
	gridNode->SetRenderable(new SceneGrid(grid));
	m_root->AddChild(gridNode);
}

void Scene::UpdateSceneGrid()
{
	SceneNode* gridNode = GetSceneGridNode();
	if (gridNode != nullptr)
	{
		SceneGrid* grid = dynamic_cast<SceneGrid*>(gridNode->GetRenderable());
		grid->SetGrid(UISettings::BuildGrid(glm::mat4(1.f)));
		gridNode->SetBBoxDirty();
		gridNode->SetCentroidDirty();
	}
}

void Scene::AddCollider(const ColliderType& type, const Vector3& center, const Vector3& param, const Vector3& velocity)
{
	SceneNode* node = new SceneNode(SceneNode::Type::SCENE_COLLIDER);

	ImplicitCollider* collider = new ImplicitCollider(type, center, param, velocity);
	SceneCollider* sceneCollider = new SceneCollider(collider);

	float mag = Vector3::Length(velocity);
	if (IsEqual(mag, 0.0f))
	{
		sceneCollider->SetVelocityMagnitude(0);
		sceneCollider->SetVelocityVector(Vector3(0, 0, 0));
	}
	else
	{
		sceneCollider->SetVelocityMagnitude(mag);
		sceneCollider->SetVelocityVector(Vector3::Normalize(velocity));
	}

	sceneCollider->UpdateMeshVelocity();

	node->SetRenderable(sceneCollider);
	glm::mat4 ctm = glm::translate(glm::mat4(1.f), glm::vec3(center));

	switch (type)
	{
	case ColliderType::SPHERE:
		ctm = glm::scale(ctm, glm::vec3(param.x));
		break;
	case ColliderType::HALF_PLANE:
		ctm *= glm::orientation(glm::vec3(param), glm::vec3(0, 1, 0));
		break;
	}

	sceneCollider->SetCTM(ctm);
	node->ApplyTransformation(ctm);
	m_root->AddChild(node);
}

void Scene::SetupLights()
{
	glm::vec4 diffuse = glm::vec4(0.5f, 0.5f, 0.5f, 1.f);
	for (int i = 0; i < 5; ++i)
	{
		glEnable(GL_LIGHT0 + i);
		glLightfv(GL_LIGHT0 + i, GL_DIFFUSE, glm::value_ptr(diffuse));
	}

	glLightfv(GL_LIGHT0, GL_POSITION, glm::value_ptr(glm::vec4(100.f, 0.f, 0.f, 1.f)));
	glLightfv(GL_LIGHT1, GL_POSITION, glm::value_ptr(glm::vec4(-100.f, 0.f, 0.f, 1.f)));
	glLightfv(GL_LIGHT2, GL_POSITION, glm::value_ptr(glm::vec4(0.f, 0.f, 100.f, 1.f)));
	glLightfv(GL_LIGHT3, GL_POSITION, glm::value_ptr(glm::vec4(0.f, 0.f, -100.f, 1.f)));
	glLightfv(GL_LIGHT4, GL_POSITION, glm::value_ptr(glm::vec4(0.f, 100.f, 0.f, 1.f)));
}