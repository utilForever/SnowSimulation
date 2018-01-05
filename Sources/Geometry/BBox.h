/*************************************************************************
> File Name: BBox.h
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Bounding box geometry of snow simulation.
> Created Time: 2018/01/02
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#ifndef SNOW_SIMULATION_BBOX_H
#define SNOW_SIMULATION_BBOX_H

#include <Common/Math.h>
#include <Common/Renderable.h>
#include <CUDA/Vector.h>
#include <Geometry/Grid.h>

#include <cmath>

class BBox : public Renderable
{
public:
	BBox();
	BBox(const BBox& other);
	BBox(const Grid& grid);
	BBox(const Vector3& p);
	BBox(const Vector3& p0, const Vector3& p1);

	void Reset();

	Vector3 GetCenter() const;
	Vector3 GetMin() const;
	Vector3 GetMax() const;

	bool IsEmpty() const;
	bool IsContains(const Vector3& point) const;
 
	Vector3 GetSize() const;
	float GetWidth() const;
	float GetHeight() const;
	float GetDepth() const;

	int GetLongestDim() const;
	float GetLongestDimSize() const;

	float GetVolume() const;
	float GetSurfaceArea() const;

	void Fix(float h);

	Grid ToGrid(float h) const;

	// Expand box by absolute distances
	void ExpandAbs(float d);
	void ExpandAbs(const Vector3& d);

	// Expand box relative to current size
	void ExpandRel(float d);
	void ExpandRel(const Vector3& d);

	// Merge two bounding boxes
	BBox& operator+=(const BBox& rhs);
	BBox operator+(const BBox& rhs) const;
	// Incorporate point into bounding box
	BBox& operator+=(const Vector3& rhs);
	BBox operator+(const Vector3& rhs) const;    
	
	void Render() override;

	BBox GetBBox(const glm::mat4& ctm) override;
	Vector3 GetCentroid(const glm::mat4& ctm) override;
	
private:
	Vector3 m_min;
	Vector3 m_max;
};

#endif