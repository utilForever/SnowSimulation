/*************************************************************************
> File Name: MitsubaExporter.cpp
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Mitsuba exporter of snow simulation.
> Created Time: 2018/01/07
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#include <Common/Math.h>
#include <Geometry/BBox.h>
#include <IO/MitsubaExporter.h>
#include <Scene/Scene.h>
#include <Scene/SceneNode.h>
#include <Simulation/Engine.h>
#include <Simulation/Particle.h>
#include <UI/UISettings.h>
#include <Viewport/Camera.h>

#include <QFile>
#include <QtConcurrent/QtConcurrent>

#include <fstream>
#include <iomanip>
#include <iostream>

#include <stdio.h>

MitsubaExporter::MitsubaExporter() : m_fps(24.f), m_filePrefix("./mts")
{
	Init();
}

MitsubaExporter::MitsubaExporter(QString fprefix, int fps) : m_fps(fps)
{
	m_filePrefix = fprefix;

	Init();
}

MitsubaExporter::~MitsubaExporter()
{
	if (m_nodes != nullptr)
	{
		delete[] m_nodes;
		m_nodes = nullptr;
	}
}

float MitsubaExporter::GetSPF()
{
	return m_spf;
}

float MitsubaExporter::GetLastUpdateTime()
{
	return m_lastUpdateTime;
}

void MitsubaExporter::Reset(Grid grid)
{
	if (m_nodes != nullptr)
	{
		delete[] m_nodes;
		m_nodes = nullptr;
	}

	m_grid = grid;
	m_nodes = new Node[m_grid.GetNumOfNodes()];
}

Node* MitsubaExporter::GetNodesPtr()
{
	return m_nodes;
}

void MitsubaExporter::RunExportThread(float t)
{
	if (m_busy == true)
	{
		m_future.waitForFinished();
	}

	m_future = QtConcurrent::run(this, &MitsubaExporter::ExportScene, t);
}

void MitsubaExporter::ExportScene(float t)
{
	m_busy = true;

	// do work here
	if (UISettings::exportDensity())
	{
		ExportDensityData(t);
	}

	if (UISettings::exportVelocity())
	{
		ExportVelocityData(t);
	}

	// colliders are written to the scene file from SceneIO because they only write once
	m_lastUpdateTime = t;
	m_frame += 1;
	m_busy = false;
}

void MitsubaExporter::ExportDensityData(float t)
{
	QString fileName = QString("%1_D_%2.vol").arg(m_filePrefix, QString("%1").arg(m_frame, 4, 'd', 0, '0'));
	std::ofstream os(fileName.toStdString().c_str());

	WriteVOLHeader(os, 1);

	int xres = m_grid.NodeDim().x;
	int yres = m_grid.NodeDim().y;
	int zres = m_grid.NodeDim().z;

	float h = m_grid.h;
	float v = h * h * h;

	for (int k = 0; k < zres; ++k)
	{
		for (int j = 0; j < yres; ++j)
		{
			for (int i = 0; i < xres; ++i)
			{
				int gIndex = (i * yres + j) * zres + k;
				float density = m_nodes[gIndex].mass / v;

				// TODO: fix this when we have more particles.
				density *= 10000;
				density = std::min(1.f, density);
				os.write(reinterpret_cast<char*>(&density), sizeof(float));
			}
		}
	}

	os.close();
}

void MitsubaExporter::ExportVelocityData(float t)
{
	QString fileName = QString("%1_V_%2.vol").arg(m_filePrefix, QString("%1").arg(m_frame, 4, 'd', 0, '0'));
	std::ofstream os(fileName.toStdString().c_str());

	WriteVOLHeader(os, 3);

	int xres = m_grid.NodeDim().x;
	int yres = m_grid.NodeDim().y;
	int zres = m_grid.NodeDim().z;

	float h = m_grid.h;
	float v = h * h * h;

	for (int k = 0; k < zres; ++k)
	{
		for (int j = 0; j < yres; ++j)
		{
			for (int i = 0; i < xres; ++i)
			{
				int gIndex = (i * yres + j) * zres + k;
				Vector3 velocity = Vector3::Min(Vector3(1), Vector3::Abs(m_nodes[gIndex].velocity));

				// RGB color channels
				for (int c = 0; c < 3; ++c) 
				{
					os.write(reinterpret_cast<char*>(&velocity[c]), sizeof(float));
				}
			}
		}
	}

	os.close();
}

void MitsubaExporter::Init()
{
	m_busy = false;
	m_nodes = nullptr;
	m_spf = 1.f / float(m_fps);
	m_lastUpdateTime = 0.f;
	m_frame = 0;
}

void MitsubaExporter::WriteVOLHeader(std::ofstream& os, int channels)
{
	int xres = m_grid.NodeDim().x;
	int yres = m_grid.NodeDim().y;
	int zres = m_grid.NodeDim().z;
	const float h = m_grid.h;

	os.write("VOL", 3);
	char version = 3;
	os.write(static_cast<char*>(&version), sizeof(char));
	int value = 1;
	os.write(reinterpret_cast<char*>(&value), sizeof(int)); //Dense float32-based representation
	os.write(reinterpret_cast<char*>(&xres), sizeof(int));
	os.write(reinterpret_cast<char*>(&yres), sizeof(int));
	os.write(reinterpret_cast<char*>(&zres), sizeof(int));
	os.write(reinterpret_cast<char*>(&channels), sizeof(int));

	// the bounding box corresponds exactly where the heterogeneous medium
	// will be positioned in MitexportVolsuba scene world space. If box is not
	// same size, stretching will occur. This is annoying when setting
	// up for arbitrary scenes, so the blender plugin will support re-writing these values
	// before rendering.
	float minX = m_grid.pos.x;
	float minY = m_grid.pos.y;
	float minZ = m_grid.pos.z;
	float maxX = minX + h * m_grid.dim.x;
	float maxY = minY + h * m_grid.dim.y;
	float maxZ = minZ + h * m_grid.dim.z;

	// bounding box
	os.write(reinterpret_cast<char*>(&minX), sizeof(float));
	os.write(reinterpret_cast<char*>(&minY), sizeof(float));
	os.write(reinterpret_cast<char*>(&minZ), sizeof(float));
	os.write(reinterpret_cast<char*>(&maxX), sizeof(float));
	os.write(reinterpret_cast<char*>(&maxY), sizeof(float));
	os.write(reinterpret_cast<char*>(&maxZ), sizeof(float));
}