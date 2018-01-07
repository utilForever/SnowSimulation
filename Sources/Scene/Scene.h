/*************************************************************************
> File Name: Scene.h
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Scene of snow simulation.
> Created Time: 2018/01/07
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#ifndef SNOW_SIMULATION_SCENE_H
#define SNOW_SIMULATION_SCENE_H

#include <Simulation/ImplicitCollider.h>

#include <glm/mat4x4.hpp>

class ParticleSystem;
class Renderable;
class SceneNode;
class SceneNodeIterator;
class QString;

class Scene
{
public:
    Scene();
    virtual ~Scene();

    virtual void Render();
    virtual void RenderVelocity(bool velTool);

    SceneNode* GetRoot();
    SceneNode* GetSceneGridNode();

    SceneNodeIterator Begin() const;

    void DeleteSelectedNodes();

    void LoadMesh(const QString& fileName, glm::mat4 ctm = glm::mat4());

    void Reset();
    void InitSceneGrid();
    void UpdateSceneGrid();

    void AddCollider(const ColliderType& type, const Vector3& center, const Vector3& param, const Vector3& velocity);

private:
    void SetupLights();

    SceneNode* m_root;
};

#endif