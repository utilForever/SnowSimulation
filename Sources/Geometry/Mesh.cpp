/*************************************************************************
> File Name: Mesh.cpp
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Mesh geometry of snow simulation.
> Created Time: 2018/01/06
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#define _USE_MATH_DEFINES
#include <cmath>

#include <Common/Util.h>
#include <CUDA/Functions.h>
#include <Geometry/BBox.h>
#include "geometry/grid.h"
#include <Geometry/Mesh.h>
#include <Simulation/ParticleSystem.h>
#include <UI/UISettings.h>

#include <GL/glew.h>
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
	m_type(Type::SNOW_CONTAINER), m_glVBO(0), m_velocityVBO(0), m_cudaVBO(nullptr),
	m_color(0.4f, 0.4f, 0.4f, 1.f), m_velVBOSize(0)
{
	// Do nothing
}

Mesh::Mesh(const QVector<Vertex>& vertices, const QVector<Tri>& tris) :
	m_type(Type::SNOW_CONTAINER), m_vertices(vertices), m_tris(tris), m_glVBO(0),
	m_velocityVBO(0), m_cudaVBO(nullptr), m_color(0.5f, 0.5f, 0.5f, 1.f), m_velVBOSize(0)
{
	// Do nothing
}

Mesh::Mesh(const QVector<Vertex>& vertices, const QVector<Tri>& tris, const QVector<Normal>& normals) :
	m_type(Type::SNOW_CONTAINER), m_vertices(vertices), m_tris(tris), m_normals(normals), m_glVBO(0),
	m_velocityVBO(0), m_cudaVBO(nullptr), m_color(0.5f, 0.5f, 0.5f, 1.f), m_velVBOSize(0)
{
	// Do nothing
}

Mesh::Mesh(const Mesh& mesh) :
	m_name(mesh.m_name), m_fileName(mesh.m_fileName), m_type(mesh.m_type),
	m_vertices(mesh.m_vertices), m_tris(mesh.m_tris), m_normals(mesh.m_normals),
	m_glVBO(0), m_velocityVBO(0), m_cudaVBO(nullptr), m_color(mesh.m_color), m_velVBOSize(0)
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

void Mesh::Render()
{
	if (!HasVBO())
	{
		BuildVBO();
	}

	if ((m_type == Type::SNOW_CONTAINER) ? UISettings::showContainers() : UISettings::showColliders())
	{

		glPushAttrib(GL_DEPTH_TEST);
		glEnable(GL_DEPTH_TEST);

		glEnable(GL_LINE_SMOOTH);
		glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

		glPushAttrib(GL_COLOR_BUFFER_BIT);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		glm::vec4 color = (m_selected) ? glm::mix(m_color, UISettings::selectionColor(), 0.5f) : m_color;

		if ((m_type == Type::SNOW_CONTAINER) ?
			(UISettings::showContainersMode() == static_cast<int>(UISettings::MeshMode::SOLID) || UISettings::showContainersMode() == static_cast<int>(UISettings::MeshMode::SOLID_AND_WIREFRAME)) :
			(UISettings::showCollidersMode() == static_cast<int>(UISettings::MeshMode::SOLID) || UISettings::showCollidersMode() == static_cast<int>(UISettings::MeshMode::SOLID_AND_WIREFRAME)))
		{
			glPushAttrib(GL_LIGHTING_BIT);
			glEnable(GL_LIGHTING);
			glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, glm::value_ptr(color*0.2f));
			glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, glm::value_ptr(color));
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			
			RenderVBO();
			
			glPopAttrib();
		}

		if ((m_type == Type::SNOW_CONTAINER) ?
			(UISettings::showContainersMode() == static_cast<int>(UISettings::MeshMode::WIREFRAME) || UISettings::showContainersMode() == static_cast<int>(UISettings::MeshMode::SOLID_AND_WIREFRAME)) :
			(UISettings::showCollidersMode() == static_cast<int>(UISettings::MeshMode::WIREFRAME) || UISettings::showCollidersMode() == static_cast<int>(UISettings::MeshMode::SOLID_AND_WIREFRAME)))
		{
			glPushAttrib(GL_POLYGON_BIT);
			glEnable(GL_POLYGON_OFFSET_LINE);
			glPolygonOffset(-1.f, -1.f);
			glPushAttrib(GL_LIGHTING_BIT);
			glDisable(GL_LIGHTING);
			glLineWidth(1.f);
			glColor4fv(glm::value_ptr(color*0.8f));
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			
			RenderVBO();
			
			glPopAttrib();
			glPopAttrib();
		}

		glPopAttrib();
		glPopAttrib();
	}
}

void Mesh::RenderForPicker()
{
	if (!HasVBO())
	{
		BuildVBO();
	}

	if ((m_type == Type::SNOW_CONTAINER) ? UISettings::showContainers() : UISettings::showColliders())
	{
		glPushAttrib(GL_DEPTH_TEST);
		glEnable(GL_DEPTH_TEST);
		glPushAttrib(GL_LIGHTING_BIT);
		glDisable(GL_LIGHTING);
		glColor3f(1.f, 1.f, 1.f);
		
		RenderVBO();
		
		glPopAttrib();
		glPopAttrib();
	}
}

void Mesh::RenderVelocityForPicker()
{
	if (!HasVelocityVBO())
	{
		BuildVelocityVBO();
	}

	if ((m_type == Type::SNOW_CONTAINER) ? UISettings::showContainers() : UISettings::showColliders())
	{
		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		
		glm::mat4 translate = glm::translate(glm::mat4(1.f), glm::vec3(GetCentroid(glm::mat4(1.f))));
		glMultMatrixf(glm::value_ptr(translate));

		glPushAttrib(GL_DEPTH_TEST);
		glEnable(GL_DEPTH_TEST);
		glPushAttrib(GL_LIGHTING_BIT);
		glDisable(GL_LIGHTING);
		glColor3f(1.f, 1.f, 1.f);

		if (!IsEqual(m_VelocityMagnitude, 0.0f))
		{
			RenderVelocityVBO();
		}

		glPopAttrib();
		glPopAttrib();

		glPopMatrix();
	}
}

void Mesh::RenderVelocity(bool velTool)
{
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();

	glm::mat4 translate = glm::translate(glm::mat4(1.f), glm::vec3(GetCentroid(glm::mat4(1.f))));
	glMultMatrixf(glm::value_ptr(translate));

	glPushAttrib(GL_DEPTH_TEST);
	glEnable(GL_DEPTH_TEST);

	glEnable(GL_LINE_SMOOTH);
	glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

	glPushAttrib(GL_COLOR_BUFFER_BIT);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glm::vec4 color = glm::vec4(.9, .9, .9, 1.0f);
	glColor4fv(glm::value_ptr(color));
	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, glm::value_ptr(color * 0.2f));
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, glm::value_ptr(color));

	if (!HasVelocityVBO())
	{
		BuildVelocityVBO();
	}

	if (velTool)
	{
		glPushAttrib(GL_DEPTH_BUFFER_BIT);
		glDisable(GL_DEPTH_TEST);
		glPushAttrib(GL_LIGHTING_BIT);
		glDisable(GL_LIGHTING);
		glPushAttrib(GL_COLOR_BUFFER_BIT);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glEnable(GL_LINE_SMOOTH);
		glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
	}

	RenderVelocityVBO();

	if (velTool)
	{
		glPopAttrib();
		glPopAttrib();
		glPopAttrib();
	}

	glPopMatrix();
}

void Mesh::UpdateMeshVelocity()
{
	DeleteVelocityVBO();
}

BBox Mesh::GetBBox(const glm::mat4& ctm)
{
	BBox box;

	for (int i = 0; i < GetNumVertices(); ++i)
	{
		const Vertex& v = m_vertices[i];
		glm::vec4 point = ctm * glm::vec4(glm::vec3(v), 1.f);

		box += Vector3(point.x, point.y, point.z);
	}

	return box;
}

Vector3 Mesh::GetCentroid(const glm::mat4& ctm)
{
	Vector3 c(0, 0, 0);

	for (int i = 0; i < GetNumVertices(); ++i)
	{
		const Vertex& v = m_vertices[i];
		glm::vec4 point = ctm * glm::vec4(glm::vec3(v), 1.f);

		c += Vector3(point.x, point.y, point.z);
	}
	return c / static_cast<float>(GetNumVertices());
}

BBox Mesh::GetObjectBBox() const
{
	BBox box;

	for (int i = 0; i < GetNumVertices(); ++i)
	{
		box += m_vertices[i];
	}

	return box;
}

bool Mesh::HasVBO() const
{
	bool has = false;

	if (m_glVBO > 0)
	{
		glBindBuffer(GL_ARRAY_BUFFER, m_glVBO);
		has = glIsBuffer(m_glVBO);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	return has;
}

void Mesh::BuildVBO()
{
	DeleteVBO();

	// Create flat array of non-indexed triangles
	Vector3* data = new Vector3[6 * GetNumTris()];
	for (int i = 0, index = 0; i < GetNumTris(); ++i)
	{
		const Tri& tri = m_tris[i];
		data[index++] = m_vertices[tri[0]];
		data[index++] = m_normals[tri[0]];
		data[index++] = m_vertices[tri[1]];
		data[index++] = m_normals[tri[1]];
		data[index++] = m_vertices[tri[2]];
		data[index++] = m_normals[tri[2]];
	}

	// Build OpenGL VBO
	glGenBuffers(1, &m_glVBO);
	glBindBuffer(GL_ARRAY_BUFFER, m_glVBO);
	glBufferData(GL_ARRAY_BUFFER, 6 * GetNumTris() * sizeof(Vector3), data, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// Register with CUDA
	RegisterVBO(&m_cudaVBO, m_glVBO);

	delete[] data;
}

void Mesh::DeleteVBO()
{
	if (m_glVBO > 0)
	{
		glBindBuffer(GL_ARRAY_BUFFER, m_glVBO);
		
		if (glIsBuffer(m_glVBO))
		{
			UnregisterVBO(m_cudaVBO);
			glDeleteBuffers(1, &m_glVBO);
		}

		glBindBuffer(GL_ARRAY_BUFFER, 0);
		m_glVBO = 0;
	}
}

void Mesh::RenderVBO()
{
	glBindBuffer(GL_ARRAY_BUFFER, m_glVBO);
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, 2 * sizeof(Vector3), static_cast<void*>(nullptr));
	glEnableClientState(GL_NORMAL_ARRAY);
	glNormalPointer(GL_FLOAT, 2 * sizeof(Vector3), reinterpret_cast<void*>(sizeof(Vector3)));

	glDrawArrays(GL_TRIANGLES, 0, 3 * GetNumTris());

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);
}

void Mesh::RenderCenter() const
{
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	
	glm::mat4 translate = glm::translate(glm::mat4(1.f), glm::vec3(0));
	glMultMatrixf(glm::value_ptr(translate));

	glBindBuffer(GL_ARRAY_BUFFER, m_velocityVBO);
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, sizeof(Vector3), static_cast<void*>(nullptr));
	glDrawArrays(GL_QUADS, m_velVBOSize - 24, 24);
	glDisableClientState(GL_VERTEX_ARRAY);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glPopMatrix();
}

void Mesh::RenderArrow()
{
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();

	Vector3 v = (this->GetCentroid(glm::mat4(1.f)));
	glm::mat4 translate = glm::translate(glm::mat4(1.f), glm::vec3(v));

	glm::mat4 basis = glm::orientation(m_velocityVector, glm::vec3(0, 1, 0));
	glMultMatrixf(glm::value_ptr(translate * basis));

	glBindBuffer(GL_ARRAY_BUFFER, m_velocityVBO);
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, sizeof(Vector3), static_cast<void*>(nullptr));
	glLineWidth(2.f);
	glDrawArrays(GL_LINES, 0, 2);
	glDrawArrays(GL_TRIANGLES, 2, m_velVBOSize - (2 + 24));
	glDisableClientState(GL_VERTEX_ARRAY);
	
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glPopMatrix();
}

bool Mesh::HasVelocityVBO() const
{
	bool has = false;

	if (m_velocityVBO > 0)
	{
		glBindBuffer(GL_ARRAY_BUFFER, m_velocityVBO);
		has = glIsBuffer(m_velocityVBO);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	return has;
}

void Mesh::BuildVelocityVBO()
{
	float scaleFactor = 4;
	QVector<Vector3> data;
	glm::vec3 pos(0);

	DeleteVBO();
	
	// Axis
	data += Vector3(pos.x, pos.y, pos.z);
	data += Vector3(pos.x, pos.y + m_VelocityMagnitude * scaleFactor, pos.z);

	// Cone
	if (m_VelocityMagnitude != 0)
	{
		static const int resolution = 60;
		static const float dTheta = 2.f * M_PI / resolution;
		static const float coneHeight = 0.1f * scaleFactor;
		static const float coneRadius = 0.05f * scaleFactor;

		for (int i = 0; i < resolution; ++i)
		{
			float upsideUp = 1;
			
			if (m_VelocityMagnitude < 0)
			{
				upsideUp = -1;
			}

			data += Vector3(pos.x, m_VelocityMagnitude * scaleFactor, pos.z);

			float theta0 = i * dTheta;
			float theta1 = (i + 1) * dTheta;

			data += (Vector3(pos.x, pos.y + m_VelocityMagnitude * scaleFactor - (upsideUp*coneHeight), pos.z) + coneRadius * Vector3(cosf(theta0), 0, -sinf(theta0)));
			data += (Vector3(pos.x, pos.y + m_VelocityMagnitude * scaleFactor - (upsideUp*coneHeight), pos.z) + coneRadius * Vector3(cosf(theta1), 0, -sinf(theta1)));
		}
	}

	// Cube
	static const float s = 0.05f;
	data += Vector3(-s, s, -s);
	data += Vector3(-s, -s, -s);
	data += Vector3(-s, -s, s);
	data += Vector3(-s, s, s);
	data += Vector3(s, s, s);
	data += Vector3(s, -s, s);
	data += Vector3(s, -s, -s);
	data += Vector3(s, s, -s);
	data += Vector3(-s, s, s);
	data += Vector3(-s, -s, s);
	data += Vector3(s, -s, s);
	data += Vector3(s, s, s);
	data += Vector3(s, s, -s);
	data += Vector3(s, -s, -s);
	data += Vector3(-s, -s, -s);
	data += Vector3(-s, s, -s);
	data += Vector3(-s, s, -s);
	data += Vector3(-s, s, s);
	data += Vector3(s, s, s);
	data += Vector3(s, s, -s);
	data += Vector3(s, s, -s);
	data += Vector3(s, s, s);
	data += Vector3(-s, s, s);
	data += Vector3(-s, s, -s);

	glGenBuffers(1, &m_velocityVBO);
	glBindBuffer(GL_ARRAY_BUFFER, m_velocityVBO);
	glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(Vector3), data.data(), GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	m_velVBOSize = data.size();
}

void Mesh::DeleteVelocityVBO()
{
	if (m_velocityVBO > 0)
	{
		glBindBuffer(GL_ARRAY_BUFFER, m_velocityVBO);

		if (glIsBuffer(m_velocityVBO))
		{
			glDeleteBuffers(1, &m_velocityVBO);
		}

		glBindBuffer(GL_ARRAY_BUFFER, 0);
		m_velocityVBO = 0;
	}
}

void Mesh::RenderVelocityVBO()
{
	float scaleConstant = .01;
	
	if (!HasVBO())
	{
		BuildVBO();
	}

	glPushAttrib(GL_DEPTH_BUFFER_BIT);
	glDisable(GL_DEPTH_TEST);
	glPushAttrib(GL_LIGHTING_BIT);
	glDisable(GL_LIGHTING);
	glPushAttrib(GL_COLOR_BUFFER_BIT);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_LINE_SMOOTH);
	glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

	glm::vec3 scaleVec;
	scaleVec.x = 1.0f / glm::length(glm::vec3(m_ctm[0][0], m_ctm[1][0], m_ctm[2][0]));
	scaleVec.y = 1.0f / glm::length(glm::vec3(m_ctm[0][1], m_ctm[1][1], m_ctm[2][1]));
	scaleVec.z = 1.0f / glm::length(glm::vec3(m_ctm[0][2], m_ctm[1][2], m_ctm[2][2]));
	
	glPushMatrix();
	
	glm::mat4 transform = glm::scale(glm::mat4(), scaleVec);
	transform = glm::scale(transform, glm::vec3(scaleConstant, scaleConstant, scaleConstant));

	glMultMatrixf(glm::value_ptr(transform));
	glLineWidth(4);
	
	RenderArrow();
	
	glLineWidth(1);
	glPopMatrix();
	
	glPopAttrib();
	glPopAttrib();
	glPopAttrib();
}