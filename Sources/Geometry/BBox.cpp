/*************************************************************************
> File Name: BBox.cpp
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Bounding box geometry of snow simulation.
> Created Time: 2018/01/05
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#include <Geometry/BBox.h>

#include <Windows.h>

#include <GL/glew.h>
#include <GL/gl.h>

#ifndef GLM_FORCE_RADIANS
#define GLM_FORCE_RADIANS
#endif

#include <glm/gtc/type_ptr.hpp>

#include <cmath>

BBox::BBox()
{
	Reset();
}

BBox::BBox(const BBox& other) : m_min(other.m_min), m_max(other.m_max)
{
	// Do nothing
}

BBox::BBox(const Grid& grid) : m_min(grid.pos), m_max(grid.pos + grid.h + Vector3(grid.dim.x, grid.dim.y, grid.dim.z))
{
	// Do nothing
}

BBox::BBox(const Vector3& p) : m_min(p), m_max(p)
{
	// Do nothing
}

BBox::BBox(const Vector3& p0, const Vector3& p1) : m_min(Vector3::Min(p0, p1)), m_max(Vector3::Max(p0, p1))
{
	// Do nothing
}

void BBox::Reset()
{
	m_min = Vector3(INFINITY, INFINITY, INFINITY);
	m_max = Vector3(-INFINITY, -INFINITY, -INFINITY);
}

Vector3 BBox::GetCenter() const
{
	return 0.5f * (m_min + m_max);
}

Vector3 BBox::GetMin() const
{
	return m_min;
}

Vector3 BBox::GetMax() const
{
	return m_max;
}

bool BBox::IsEmpty() const
{
	return m_min.x > m_max.x;
}

bool BBox::IsContains(const Vector3& point) const
{
	return
		(point.x >= m_min.x && point.x <= m_max.x) &&
		(point.y >= m_min.y && point.y <= m_max.y) &&
		(point.z >= m_min.z && point.z <= m_max.z);
}

Vector3 BBox::GetSize() const
{
	return m_max - m_min;
}

float BBox::GetWidth() const
{
	return m_max.x - m_min.x;
}

float BBox::GetHeight() const
{
	return m_max.y - m_min.y;
}

float BBox::GetDepth() const
{
	return m_max.z - m_min.z;
}

int BBox::GetLongestDim() const
{
	Vector3 size = m_max - m_min;
	return (size.x > size.y) ? ((size.x > size.z) ? 0 : 2) : ((size.y > size.z) ? 1 : 2);
}

float BBox::GetLongestDimSize() const
{
	Vector3 size = m_max - m_min;
	return (size.x > size.y) ? ((size.x > size.z) ? size.x : size.z) : ((size.y > size.z) ? size.y : size.z);
}

float BBox::GetVolume() const
{
	Vector3 s = m_max - m_min;
	return s.x * s.y * s.z;
}

float BBox::GetSurfaceArea() const
{
	Vector3 s = m_max - m_min;
	return 2 * (s.x * s.y + s.y * s.z + s.z * s.x);
}

void BBox::Fix(float h)
{
	Vector3 c = 0.5f * (m_min + m_max);
	Vector3 d = h * Vector3::Ceil((m_max - m_min) / h) / 2.f;
	
	m_min = c - d;
	m_max = c + d;
}

Grid BBox::ToGrid(float h) const
{
	BBox box(*this);
	box.ExpandAbs(h);
	box.Fix(h);
	
	Grid grid;
	Vector3 dimf = Vector3::Round((box.GetMax() - box.GetMin()) / h);
	grid.dim = glm::ivec3(dimf.x, dimf.y, dimf.z);
	grid.h = h;
	grid.pos = box.GetMin();
	
	return grid;
}

void BBox::ExpandAbs(float d)
{
	m_min -= Vector3(d, d, d);
	m_max += Vector3(d, d, d);
}

void BBox::ExpandAbs(const Vector3& d)
{
	m_min -= d;
	m_max += d;
}

void BBox::ExpandRel(float d)
{
	Vector3 dd = d * (m_max - m_min);
	m_min -= dd;
	m_max += dd;
}

void BBox::ExpandRel(const Vector3& d)
{
	Vector3 dd = d * (m_max - m_min);
	m_min -= dd;
	m_max += dd;
}

BBox& BBox::operator+=(const BBox& rhs)
{
	m_min = Vector3::Min(m_min, rhs.m_min);
	m_max = Vector3::Max(m_max, rhs.m_max);
	
	return *this;
}

BBox BBox::operator+(const BBox& rhs) const
{
	return BBox(Vector3::Min(m_min, rhs.m_min), Vector3::Max(m_max, rhs.m_max));
}

BBox& BBox::operator+=(const Vector3& rhs)
{
	m_min = Vector3::Min(m_min, rhs);
	m_max = Vector3::Max(m_max, rhs);

	return *this;
}

BBox BBox::operator+(const Vector3& rhs) const
{
	return BBox(Vector3::Min(m_min, rhs), Vector3::Max(m_max, rhs));
}

void BBox::Render()
{
	glm::vec3 corners[8];
	glm::vec3 corner;

	for (size_t x = 0, index = 0; x <= 1; ++x)
	{
		corner.x = (x ? m_max : m_min).x;

		for (size_t y = 0; y <= 1; ++y)
		{
			corner.y = (y ? m_max : m_min).y;
		
			for (size_t z = 0; z <= 1; ++z, ++index)
			{
				corner.z = (z ? m_max : m_min).z;
				corners[index] = corner;
			}
		}
	}

	glBegin(GL_LINES);

	glVertex3fv(glm::value_ptr(corners[0]));
	glVertex3fv(glm::value_ptr(corners[1]));

	glVertex3fv(glm::value_ptr(corners[1]));
	glVertex3fv(glm::value_ptr(corners[3]));

	glVertex3fv(glm::value_ptr(corners[3]));
	glVertex3fv(glm::value_ptr(corners[2]));

	glVertex3fv(glm::value_ptr(corners[2]));
	glVertex3fv(glm::value_ptr(corners[0]));

	glVertex3fv(glm::value_ptr(corners[2]));
	glVertex3fv(glm::value_ptr(corners[6]));

	glVertex3fv(glm::value_ptr(corners[3]));
	glVertex3fv(glm::value_ptr(corners[7]));

	glVertex3fv(glm::value_ptr(corners[1]));
	glVertex3fv(glm::value_ptr(corners[5]));

	glVertex3fv(glm::value_ptr(corners[0]));
	glVertex3fv(glm::value_ptr(corners[4]));

	glVertex3fv(glm::value_ptr(corners[6]));
	glVertex3fv(glm::value_ptr(corners[7]));

	glVertex3fv(glm::value_ptr(corners[7]));
	glVertex3fv(glm::value_ptr(corners[5]));

	glVertex3fv(glm::value_ptr(corners[5]));
	glVertex3fv(glm::value_ptr(corners[4]));

	glVertex3fv(glm::value_ptr(corners[4]));
	glVertex3fv(glm::value_ptr(corners[6]));

	glEnd();
}

BBox BBox::GetBBox(const glm::mat4& ctm)
{
	BBox box;
	Vector3 corner;

	for (size_t x = 0, index = 0; x <= 1; ++x)
	{
		corner.x = (x ? m_max : m_min).x;
		for (size_t y = 0; y <= 1; ++y)
		{
			corner.y = (y ? m_max : m_min).y;
			for (size_t z = 0; z <= 1; ++z, ++index)
			{
				corner.z = (z ? m_max : m_min).z;

				glm::vec4 point = ctm * glm::vec4(corner.x, corner.y, corner.z, 1.f);
				box += Vector3(point.x, point.y, point.z);
			}
		}
	}

	return box;
}

Vector3 BBox::GetCentroid(const glm::mat4& ctm)
{
	Vector3 c = GetCenter();
	glm::vec4 p = ctm * glm::vec4(c.x, c.y, c.z, 1.f);

	return Vector3(p.x, p.y, p.z);
}