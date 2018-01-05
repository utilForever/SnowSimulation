/*************************************************************************
> File Name: Mesh.h
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Mesh geometry of snow simulation.
> Created Time: 2018/01/05
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#ifndef SNOW_SIMULATION_MESH_H
#define SNOW_SIMULATION_MESH_H

#include <Common/Renderable.h>
#include <UI/UISettings.h>

#ifndef GLM_FORCE_RADIANS
#define GLM_FORCE_RADIANS
#endif

#include <glm/mat4x4.hpp>

#include <QVector>
#include <QString>

using GLuint = unsigned int;
using Vertex = Vector3;
using Normal = Vector3;
using Color = glm::vec4;

struct cudaGraphicsResource;

class ParticleSystem;
class BBox;

class Mesh : public Renderable
{
public:
	enum class Type
	{
		SNOW_CONTAINER,
		COLLIDER
	};

	struct Tri
	{
		
	};

private:
	QString m_name;
	// The OBJ file source
	QString m_fileName;
	Type m_type;

	// List of vertices
	QVector<Vertex> m_vertices;

	// List of tris, which index into vertices
	QVector<Tri> m_tris;

	// List of vertex normals
	QVector<Normal> m_normals;

	// OpenGL stuff
	GLuint m_glVBO, m_velVBO;
	cudaGraphicsResource *m_cudaVBO;

	Color m_color;

	int m_velVBOSize;

	bool HasVBO() const;
	void BuildVBO();
	void DeleteVBO();

	void RenderVBO();
	void RenderCenter() const;
	void RenderArrow();

	bool HasVelVBO() const;
	void BuildVelVBO();
	void DeleteVelVBO();

	void RenderVelVBO();
};

#endif