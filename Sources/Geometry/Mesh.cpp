/*************************************************************************
> File Name: Mesh.cpp
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Mesh geometry of snow simulation.
> Created Time: 2018/01/06
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#include <Common/Util.h>
#include <CUDA/Functions.h>
#include <Geometry/BBox.h>
#include "geometry/grid.h"
#include <Geometry/Mesh.h>
#include <Simulation/ParticleSystem.h>
#include <UI/UISettings.h>
#include <UI/Tools/Tool.h>

#include <GL/gl.h>

#ifndef GLM_FORCE_RADIANS
#define GLM_FORCE_RADIANS
#endif

#include <glm/geometric.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <glm/gtx/string_cast.hpp>

#include <QElapsedTimer>
#include <QLocale>

Mesh::Mesh() :
	m_type(Type::SNOW_CONTAINER), m_glVBO(0), m_velVBO(0), m_cudaVBO(nullptr),
	m_color(0.4f, 0.4f, 0.4f, 1.f), m_velVBOSize(0)
{
	// Do nothing
}

Mesh::Mesh(const QVector<Vertex>& vertices, const QVector<Tri>& tris) :
	m_type(Type::SNOW_CONTAINER), m_vertices(vertices), m_tris(tris), m_glVBO(0),
	m_velVBO(0), m_cudaVBO(nullptr), m_color(0.5f, 0.5f, 0.5f, 1.f), m_velVBOSize(0)
{
	// Do nothing
}

Mesh::Mesh(const QVector<Vertex>& vertices, const QVector<Tri>& tris, const QVector<Normal>& normals) :
	m_type(Type::SNOW_CONTAINER), m_vertices(vertices), m_tris(tris), m_normals(normals), m_glVBO(0),
	m_velVBO(0), m_cudaVBO(nullptr), m_color(0.5f, 0.5f, 0.5f, 1.f), m_velVBOSize(0)
{
	// Do nothing
}

Mesh::Mesh(const Mesh& mesh) :
	m_name(mesh.m_name), m_fileName(mesh.m_fileName), m_type(mesh.m_type),
	m_vertices(mesh.m_vertices), m_tris(mesh.m_tris), m_normals(mesh.m_normals),
	m_glVBO(0), m_velVBO(0), m_cudaVBO(nullptr), m_color(mesh.m_color), m_velVBOSize(0)
{
	// Do nothing
}

Mesh::~Mesh()
{
	DeleteVBO();
}

void Mesh::SetType(Type type)
{
	m_type = type;
}

Mesh::Type Mesh::GetType() const
{
	return m_type;
}

void Mesh::Fill(ParticleSystem& particles, int particleCount, float h, float targetDensity, int materialPreset)
{
	if (!HasVBO())
	{
		BuildVBO();
	}

	QElapsedTimer timer;
	timer.start();

	Grid grid = GetObjectBBox().ToGrid(h);

	LOG("Filling mesh in %d x %d x %d grid (%s voxels)...", grid.dim.x, grid.dim.y, grid.dim.z, STR(QLocale().toString(grid.dim.x * grid.dim.y * grid.dim.z)));

	particles.Resize(particleCount);
	FillMesh(&m_cudaVBO, GetNumTris(), grid, particles.GetData(), particleCount, targetDensity, materialPreset);

	LOG("Mesh filled with %s particles in %lld ms.", STR(QLocale().toString(particleCount)), timer.restart());
}

bool Mesh::IsEmpty() const
{
	return m_vertices.empty() || m_tris.empty();
}

void Mesh::Clear()
{
	m_vertices.clear();
	m_tris.clear();
	m_normals.clear();

	DeleteVBO();
}

void Mesh::ApplyTransformation(const glm::mat4& transform)
{
	for (int i = 0; i < GetNumVertices(); ++i)
	{
		const Vertex& v = m_vertices[i];
		glm::vec4 point = transform * glm::vec4(glm::vec3(v), 1.f);

		m_vertices[i] = Vector3(point.x, point.y, point.z);
	}

	ComputeNormals();
	DeleteVBO();
}

void Mesh::Append(const Mesh& mesh)
{
	int offset = m_vertices.size();
	
	for (int i = 0; i < mesh.m_tris.size(); ++i)
	{
		Tri tri = mesh.m_tris[i];
		tri.Offset(offset);
		m_tris += tri;
	}

	m_vertices += mesh.m_vertices;
	m_normals += mesh.m_normals;
	
	DeleteVBO();
}

void Mesh::ComputeNormals()
{
	Normal* triNormals = new Normal[GetNumTris()];
	float* triAreas = new float[GetNumTris()];
	QVector<int>* vertexMembership = new QVector<int>[GetNumVertices()];

	for (int i = 0; i < GetNumTris(); ++i)
	{
		// Compute triangle normal and area
		const Tri &tri = m_tris[i];
		const Vertex &v0 = m_vertices[tri[0]];
		const Vertex &v1 = m_vertices[tri[1]];
		const Vertex &v2 = m_vertices[tri[2]];
		Normal n = Vector3::Cross(v1 - v0, v2 - v0);

		triAreas[i] = Vector3::Length(n) / 2.f;
		triNormals[i] = 2.f * n / triAreas[i];

		// Record triangle membership for each vertex
		vertexMembership[tri[0]] += i;
		vertexMembership[tri[1]] += i;
		vertexMembership[tri[2]] += i;
	}

	m_normals.clear();
	m_normals.resize(GetNumVertices());

	for (int i = 0; i < GetNumVertices(); ++i)
	{
		Normal normal = Normal(0.f, 0.f, 0.f);
		float sum = 0.f;

		for (int j = 0; j < vertexMembership[i].size(); ++j)
		{
			int index = vertexMembership[i][j];

			normal += triAreas[index] * triNormals[index];
			sum += triAreas[index];
		}

		normal /= sum;
		m_normals[i] = normal;
	}

	delete[] triNormals;
	delete[] triAreas;
	delete[] vertexMembership;
}

void Mesh::SetName(const QString& name)
{
	m_name = name;
}

QString Mesh::GetName() const
{
	return m_name;
}

void Mesh::SetFileName(const QString& fileName)
{
	m_fileName = fileName;
}

QString Mesh::GetFileName() const
{
	return m_fileName;
}

void Mesh::SetVertices(const QVector<Vertex>& vertices)
{
	m_vertices = vertices;
}

void Mesh::AddVertex(const Vertex& vertex)
{
	m_vertices += vertex;
}

int Mesh::GetNumVertices() const
{
	return m_vertices.size();
}

Vertex& Mesh::GetVertex(int i)
{
	return m_vertices[i];
}

Vertex Mesh::GetVertex(int i) const
{
	return m_vertices[i];
}

QVector<Vertex>& Mesh::GetVertices()
{
	return m_vertices;
}

const QVector<Vertex>& Mesh::GetVertices() const
{
	return m_vertices;
}

void Mesh::SetTris(const QVector<Tri>& tris)
{
	m_tris = tris;
}

void Mesh::AddTri(const Tri& tri)
{
	m_tris += tri;
}

int Mesh::GetNumTris() const
{
	return m_tris.size();
}

Mesh::Tri& Mesh::GetTri(int i)
{
	return m_tris[i];
}

Mesh::Tri Mesh::GetTri(int i) const
{
	return m_tris[i];
}

QVector<Mesh::Tri>& Mesh::GetTris()
{
	return m_tris;
}
const QVector<Mesh::Tri>& Mesh::GetTris() const
{
	return m_tris;
}

void Mesh::SetNormals(const QVector<Normal>& normals)
{
	m_normals = normals;
}

void Mesh::AddNormal(const Normal& normal)
{
	m_normals += normal;
}

int Mesh::GetNumNormals() const
{
	return m_normals.size();
}

Normal& Mesh::GetNormal(int i)
{
	return m_normals[i];
}

Normal Mesh::GetNormal(int i) const
{
	return m_normals[i];
}

QVector<Normal>& Mesh::GetNormals()
{
	return m_normals;
}

const QVector<Normal>& Mesh::GetNormals() const
{
	return m_normals;
}