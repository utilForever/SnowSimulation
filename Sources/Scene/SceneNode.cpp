/*************************************************************************
> File Name: SceneNode.cpp
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Scene node of snow simulation.
> Created Time: 2018/01/07
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#include <Common/Renderable.h>
#include <Scene/SceneCollider.h>
#include <Scene/SceneNode.h>

#include <GL/glew.h>
#include <GL/gl.h>

#ifndef GLM_FORCE_RADIANS
#define GLM_FORCE_RADIANS
#endif

#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/type_ptr.hpp>

SceneNode::SceneNode(Type type) :
    m_parent(nullptr), m_ctm(1.f), m_ctmDirty(true), m_bboxDirty(true), m_centroidDirty(true),
    m_transform(1.f), m_renderable(nullptr), m_type(type)
{
    // Do nothing
}

SceneNode::~SceneNode()
{
    if (m_renderable != nullptr)
    {
        delete m_renderable;
        m_renderable = nullptr;
    }

    ClearChild();
}

void SceneNode::ClearChild()
{
    for (int i = 0; i < m_child.size(); ++i)
    {
        if (m_child[i] != nullptr)
        {
            delete m_child[i];
            m_child[i] = nullptr;
        }
    }

    m_child.clear();
}

void SceneNode::AddChild(SceneNode* child)
{
    m_child += child;
    child->m_parent = this;
    child->SetCTMDirty();
}

void SceneNode::DeleteChild(SceneNode* child)
{
    int index = m_child.indexOf(child);
    if (index != -1)
    {
        SceneNode* child = m_child[index];
        
        if (child != nullptr)
        {
            delete child;
            child = nullptr;
        }

        m_child.removeAt(index);
    }
}

SceneNode* SceneNode::GetParent()
{
    return m_parent;
}

QList<SceneNode*> SceneNode::GetChild()
{
    return m_child;
}

bool SceneNode::HasRenderable() const
{
    return m_renderable != nullptr;
}

void SceneNode::SetRenderable(Renderable* renderable)
{
    if (m_renderable != nullptr)
    {
        delete m_renderable;
        m_renderable = nullptr;
    }

    m_renderable = renderable;
}

Renderable* SceneNode::GetRenderable()
{
    return m_renderable;
}

void SceneNode::RenderOpaque()
{
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glMultMatrixf(glm::value_ptr(GetCTM()));

    if (m_renderable != nullptr && IsTransparent() == false)
    {
        m_renderable->Render();
    }

    glPopMatrix();

    for (int i = 0; i < m_child.size(); ++i)
    {
        m_child[i]->RenderOpaque();
    }
}

void SceneNode::RenderTransparent()
{
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glMultMatrixf(glm::value_ptr(GetCTM()));

    if (m_renderable != nullptr && IsTransparent() == true)
    {
        m_renderable->Render();
    }

    glPopMatrix();

    for (int i = 0; i < m_child.size(); ++i)
    {
        m_child[i]->RenderTransparent();
    }
}

void SceneNode::RenderVelocity(bool velTool)
{
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glMultMatrixf(glm::value_ptr(GetCTM()));

    if (m_renderable != nullptr)
    {
        m_renderable->RenderVelocity(velTool);
    }

    glPopMatrix();

    for (int i = 0; i < m_child.size(); ++i)
    {
        m_child[i]->RenderVelocity(velTool);
    }
}

glm::mat4 SceneNode::GetCTM()
{
    if (m_ctmDirty == true)
    {
        glm::mat4 pCtm = (m_parent) ? m_parent->GetCTM() : glm::mat4();
        m_ctm = pCtm * m_transform;
        m_ctmDirty = false;
    }

    return m_ctm;
}

void SceneNode::SetCTMDirty()
{
    for (int i = 0; i < m_child.size(); ++i)
    {
        m_child[i]->SetCTMDirty();
    }

    m_ctmDirty = true;
    m_bboxDirty = true;
    m_centroidDirty = true;
}

void SceneNode::ApplyTransformation(const glm::mat4& transform)
{
    m_transform = transform * m_transform;
    SetCTMDirty();

    if (this->HasRenderable() == true)
    {
        GetCTM();
        this->GetRenderable()->SetCTM(m_ctm);
    }
}

BBox SceneNode::GetBBox()
{
    if (m_bboxDirty == true || GetType() == Type::SCENE_COLLIDER)
    {
        if (HasRenderable() == true)
        {
            m_bbox = m_renderable->GetBBox(GetCTM());
        }
        else
        {
            m_bbox = BBox();
        }

        m_bboxDirty = false;
    }

    return m_bbox;
}

void SceneNode::SetBBoxDirty()
{
    m_bboxDirty = true;
}

Vector3 SceneNode::GetCentroid()
{
    if (m_centroidDirty == true)
    {
        if (HasRenderable() == true)
        {
            m_centroid = m_renderable->GetCentroid(GetCTM());
        }
        else
        {
            glm::vec4 p = GetCTM() * glm::vec4(0, 0, 0, 1);
            m_centroid = Vector3(p.x, p.y, p.z);
        }

        m_centroidDirty = false;
    }

    return m_centroid;
}

void SceneNode::SetCentroidDirty()
{
    m_centroidDirty = true;
}

SceneNode::Type SceneNode::GetType()
{
    return m_type;
}

bool SceneNode::IsTransparent() const
{
    return m_type == Type::SIMULATION_GRID;
}