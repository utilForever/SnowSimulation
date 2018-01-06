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
struct BBox;

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
        union
	    {
            struct { int a, b, c; };
            int corners[3];
        };

        Tri() : a(-1), b(-1), c(-1)
        {
            // Do nothing
        }

        Tri(int i0, int i1, int i2) : a(i0), b(i1), c(i2)
        {
            // Do nothing
        }

        Tri(const Tri& other) : a(other.a), b(other.b), c(other.c)
        {
            // Do nothing
        }

        void Reverse()
        {
            std::swap(a, c);
        }

        void Offset(int offset)
        {
            a += offset;
            b += offset;
            c += offset;
        }

        int& operator[](int i)
        {
            return corners[i];
        }
        
	    int operator[](int i) const
        {
            return corners[i];
        }
	};

    Mesh();
    Mesh(const QVector<Vertex>& vertices, const QVector<Tri>& tris);
    Mesh(const QVector<Vertex>& vertices, const QVector<Tri>& tris, const QVector<Normal>& normals);
    Mesh(const Mesh& mesh);

    virtual ~Mesh();

    void SetType(Type type);
    Type GetType() const;

    void Fill(ParticleSystem& particles, int particleCount, float h, float targetDensity, int materialPreset);

    bool IsEmpty() const;
    void Clear();

    void ApplyTransformation(const glm::mat4& transform);

    void Append(const Mesh& mesh);

    void ComputeNormals();

    void SetName(const QString& name);
    QString GetName() const;

    void SetFileName(const QString& fileName);
    QString GetFileName() const;

    void SetVertices(const QVector<Vertex>& vertices);
    void AddVertex(const Vertex& vertex);
    int GetNumVertices() const;
    Vertex& GetVertex(int i);
    Vertex GetVertex(int i) const;
    QVector<Vertex>& GetVertices();
    const QVector<Vertex>& GetVertices() const;

    void SetTris(const QVector<Tri>& tris);
    void AddTri(const Tri& tri);
    int GetNumTris() const;
    Tri& GetTri(int i);
    Tri GetTri(int i) const;
    QVector<Tri>& GetTris();
    const QVector<Tri>& GetTris() const;

    void SetNormals(const QVector<Normal>& normals);
    void AddNormal(const Normal& normal);
    int GetNumNormals() const;
    Normal& GetNormal(int i);
    Normal GetNormal(int i) const;
    QVector<Normal>& GetNormals();
    const QVector<Normal>& GetNormals() const;

    void Render() override;
    void RenderForPicker() override;
    void RenderVelocityForPicker() override;

    void RenderVelocity(bool velTool) override;

    void UpdateMeshVelocity() override;

    BBox GetBBox(const glm::mat4& ctm) override;
    Vector3 GetCentroid(const glm::mat4& ctm) override;

    BBox GetObjectBBox() const;

private:
    bool HasVBO() const;
    void BuildVBO();
    void DeleteVBO();

    void RenderVBO();
    void RenderCenter() const;
    void RenderArrow();

    bool HasVelocityVBO() const;
    void BuildVelocityVBO();
    void DeleteVelocityVBO();

    void RenderVelocityVBO();

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
	GLuint m_glVBO, m_velocityVBO;
	cudaGraphicsResource* m_cudaVBO;

	Color m_color;

	int m_velVBOSize;
};

#endif