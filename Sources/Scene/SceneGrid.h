/*************************************************************************
> File Name: SceneGrid.h
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Scene grid of snow simulation.
> Created Time: 2018/01/07
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#ifndef SNOW_SIMULATION_SCENE_GRID_H
#define SNOW_SIMULATION_SCENE_GRID_H

#include <Common/Renderable.h>
#include <Geometry/Grid.h>

using GLuint = unsigned int;

class SceneGrid : public Renderable
{
public:
	SceneGrid();
	SceneGrid(const Grid& grid);
	virtual ~SceneGrid();

	void Render() override;
	void RenderForPicker() override;

	BBox GetBBox(const glm::mat4& ctm) override;
	Vector3 GetCentroid(const glm::mat4& ctm) override;

	void SetGrid(const Grid& grid);

private:
	bool HasVBO() const;
	void BuildVBO();
	void DeleteVBO();

	Grid m_grid;

	GLuint m_vbo;
	int m_vboSize;
};

#endif