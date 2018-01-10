/*************************************************************************
> File Name: SelectionTool.cpp
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Selection tool of snow simulation.
> Created Time: 2018/01/10
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#include <Common/Renderable.h>
#include <CUDA/Vector.h>
#include <Geometry/BBox.h>
#include <Scene/Scene.h>
#include <Scene/SceneNode.h>
#include <Scene/SceneNodeIterator.h>
#include <UI/Picker.h>
#include <UI/UserInput.h>
#include <UI/ViewPanel.h>
#include <UI/Tools/SelectionTool.h>
#include <Viewport/Viewport.h>

#ifndef GLM_FORCE_RADIANS
#define GLM_FORCE_RADIANS
#endif

#include <glm/mat4x4.hpp>
#include <glm/gtc/type_ptr.hpp>

SelectionTool::SelectionTool(ViewPanel* panel, Type t) : Tool(panel, t)
{
    // Do nothing
}

SelectionTool::~SelectionTool()
{
    // Do nothing
}

void SelectionTool::MousePressed()
{
    Tool::MousePressed();
}

void SelectionTool::MouseReleased()
{
    if (m_mouseDown == true)
    {
        SceneNode* selected = GetSelectedSceneNode();

        if (UserInput::IsShiftKey() == true)
        {
            if (selected != nullptr)
            {
                selected->GetRenderable()->SetSelected(!selected->GetRenderable()->GetSelected());
            }
        }
        else
        {
            ClearSelection();
            
            if (selected != nullptr)
            {
                selected->GetRenderable()->SetSelected(true);
            }
        }
        
        Tool::MouseReleased();
    }

    m_panel->CheckSelected();
}

void SelectionTool::Update()
{
    // Do nothing
}

void SelectionTool::Render()
{
    // Do nothing
}

bool
SelectionTool::HasSelection(Vector3& center) const
{
    center = Vector3(0, 0, 0);
    int count = 0;

    for (SceneNodeIterator iter = m_panel->m_scene->Begin(); iter.IsValid(); ++iter)
    {
        if ((*iter)->HasRenderable() == true && (*iter)->GetRenderable()->GetSelected() == true)
        {
            center += (*iter)->GetCentroid();
            count++;
        }
    }

    center /= static_cast<float>(count);
    return count > 0;
}

bool SelectionTool::HasRotatableSelection(Vector3& center) const
{
    center = Vector3(0, 0, 0);
    int count = 0;

    for (SceneNodeIterator iter = m_panel->m_scene->Begin(); iter.IsValid(); ++iter)
    {
        if ((*iter)->HasRenderable() == true && (*iter)->GetRenderable()->GetSelected() == true &&
            (*iter)->GetType() != SceneNode::Type::SIMULATION_GRID)
        {
            center += (*iter)->GetCentroid();
            count++;
        }
    }

    center /= static_cast<float>(count);
    return count > 0;
}

bool SelectionTool::HasScalableSelection(Vector3& center) const
{
    return HasRotatableSelection(center);
}

void SelectionTool::ClearSelection()
{
    m_panel->ClearSelection();
}

SceneNode* SelectionTool::GetSelectedSceneNode()
{
    m_panel->m_viewport->LoadPickMatrices(UserInput::GetMousePos(), 3.f);

    SceneNode* clicked = nullptr;

    QList<SceneNode*> renderables;
    for (SceneNodeIterator iter = m_panel->m_scene->Begin(); iter.IsValid(); ++iter)
    {
        if ((*iter)->HasRenderable())
        {
            renderables += *iter;
        }
    }

    if (renderables.empty() == false)
    {
        Picker picker(renderables.size());
        for (int i = 0; i < renderables.size(); ++i)
        {
            glMatrixMode(GL_MODELVIEW);
            glPushMatrix();
            glMultMatrixf(glm::value_ptr(renderables[i]->GetCTM()));
            
            picker.SetObjectIndex(i);
            renderables[i]->GetRenderable()->RenderForPicker();
            
            glPopMatrix();
        }

        unsigned int index = picker.GetPick();
        if (index != Picker::NO_PICK)
        {
            clicked = renderables[index];
        }
    }

    m_panel->m_viewport->PopMatrices();

    return clicked;
}