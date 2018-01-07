/*************************************************************************
> File Name: SceneNode.h
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Scene node of snow simulation.
> Created Time: 2018/01/07
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#ifndef SNOW_SIMULATION_SCENE_NODE_H
#define SNOW_SIMULATION_SCENE_NODE_H

#include <Geometry/BBox.h>

#ifndef GLM_FORCE_RADIANS
#define GLM_FORCE_RADIANS
#endif

#include <glm/mat4x4.hpp>

#include <QList>
#include <QString>

class Renderable;

class SceneNode
{
public:

	enum class Type
	{
		TRANSFORM,
		SCENE_COLLIDER,
		SNOW_CONTAINER,
		SIMULATION_GRID
	};

	SceneNode(Type type = Type::TRANSFORM);
	virtual ~SceneNode();

	void ClearChild();
	void AddChild(SceneNode* child);
	// A scene node should be deleted through its parent using this
	// function (unless it's the root node) so that the parent
	// doesn't have a dangling NULL pointer
	void DeleteChild(SceneNode* child);

	SceneNode* GetParent();
	QList<SceneNode*> GetChild();

	bool HasRenderable() const;
	void SetRenderable(Renderable* renderable);
	Renderable* GetRenderable();

	// Render the node's renderable if it is opaque
	virtual void RenderOpaque();
	// Render the node's renderable if it is transparent
	virtual void RenderTransparent();

	virtual void RenderVelocity(bool velTool);

	glm::mat4 GetCTM();
	// Indicate that the CTM needs recomputing
	void SetCTMDirty();

	void ApplyTransformation(const glm::mat4& transform);

	// World space bounding box
	BBox GetBBox();
	// Indicate that the world space bounding box needs recomputing
	void SetBBoxDirty();

	// World space centroid
	Vector3 GetCentroid();
	// Indicate that the world space centroid needs recomputing
	void SetCentroidDirty();

	Type GetType();

	// For now, only scene grid nodes are transparent;
	bool IsTransparent() const;

private:
	SceneNode * m_parent;

	// The following member variables depend on the scene node's
	// cumulative transformation, so they are cached and only
	// recomputed when necessary, if they are labeled "dirty".
	glm::mat4 m_ctm;
	bool m_ctmDirty;
	BBox m_bbox;
	bool m_bboxDirty;
	Vector3 m_centroid;
	bool m_centroidDirty;

	glm::mat4 m_transform;

	QList<SceneNode*> m_child;
	Renderable* m_renderable;

	Type m_type;
};

#endif