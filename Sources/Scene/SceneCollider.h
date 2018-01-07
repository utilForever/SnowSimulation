/*************************************************************************
> File Name: SceneCollider.h
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Scene collider of snow simulation.
> Created Time: 2018/01/07
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#ifndef SNOW_SIMULATION_SCENE_COLLIDER_H
#define SNOW_SIMULATION_SCENE_COLLIDER_H

#include <Common/Renderable.h>

struct BBox;
struct Mesh;
struct ImplicitCollider;

class SceneCollider : public Renderable
{
public:
	SceneCollider(ImplicitCollider* collider);
	virtual ~SceneCollider();

	void Render() override;
	void RenderForPicker() override;
	void RenderVelocityForPicker() override;
	void UpdateMeshVelocity() override;

	static constexpr float GetSphereRadius();

	BBox GetBBox(const glm::mat4& ctm) override;
	Vector3 GetCentroid(const glm::mat4& ctm) override;

	void InitializeMesh();

	void SetSelected(bool selected) override;

	void SetCTM(const glm::mat4& ctm) override;

	void RenderVelocity(bool velTool) override;

	ImplicitCollider* GetImplicitCollider();

private:
	ImplicitCollider* m_collider;
	Mesh* m_mesh;
};

#endif