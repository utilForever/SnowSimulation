/*************************************************************************
> File Name: UISettings.h
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: UI settings of snow simulation.
> Created Time: 2017/12/28
> Copyright (c) 2017, Chan-Ho Chris Ohk
*************************************************************************/
#ifndef SNOW_SIMULATION_UI_SETTINGS_H
#define SNOW_SIMULATION_UI_SETTINGS_H

#define DEFINE_SETTING(TYPE, NAME)                                  \
	private:                                                        \
		TYPE m_##NAME;                                              \
	public:                                                         \
		static TYPE& NAME() { return GetInstance()->m_##NAME; }     \

#include <QPoint>
#include <QSize>
#include <QVariant>

#ifndef GLM_FORCE_RADIANS
#define GLM_FORCE_RADIANS
#endif

#include "glm/vec3.hpp"
#include "glm/vec4.hpp"
#include "glm/mat4x4.hpp"

#include "CUDA/vector.h"

// Forward declaration
struct Grid;

class UISettings
{
public:
	enum class MeshMode
	{
		WIREFRAME,
		SOLID,
		SOLID_AND_WIREFRAME
	};

	enum class GridMode
	{
		BOX,
		MIN_FACE_CELLS,
		ALL_FACE_CELLS
	};

	enum class GridDataMode
	{
		NODE_DENSITY,
		NODE_VELOCITY,
		NODE_SPEED,
		NODE_FORCE
	};

	enum ParticlesMode
	{
		PARTICLE_MASS,
		PARTICLE_VELOCITY,
		PARTICLE_SPEED,
		PARTICLE_STIFFNESS
	};

	enum class SnowMaterialPreset
	{
		MAT_DEFAULT,
		MAT_CHUNKY
	};

	static UISettings* GetInstance();
	static void DeleteInstance();

	static QVariant GetSetting(const QString& name, const QVariant& value = QVariant());
	static void SetSetting(const QString& name, const QVariant& value);

	static void LoadSettings();
	static void SaveSettings();

	static Grid BuildGrid(const glm::mat4& ctm);

protected:
	UISettings();
	virtual ~UISettings();

private:
	static UISettings* m_instance;

	DEFINE_SETTING(QPoint, m_windowPosition);
	DEFINE_SETTING(QSize, m_windowSize);

	// Filling
	DEFINE_SETTING(int, m_fillNumParticles);
	DEFINE_SETTING(float, m_fillDensity);
	DEFINE_SETTING(float, m_fillResolution);

	// Exporting
	DEFINE_SETTING(bool, m_exportDensity);
	DEFINE_SETTING(bool, m_exportVelocity);
	DEFINE_SETTING(int, m_exportFPS);
	DEFINE_SETTING(float, m_maxTime);

	DEFINE_SETTING(vec3, m_gridPosition);
	DEFINE_SETTING(glm::ivec3, m_gridDimensions);
	DEFINE_SETTING(float, m_gridResolution);

	DEFINE_SETTING(float, m_timeStep);
	DEFINE_SETTING(bool, m_implicit);
	DEFINE_SETTING(int, m_materialPreset);

	DEFINE_SETTING(bool, m_showContainers);
	DEFINE_SETTING(int, m_showContainersMode);
	DEFINE_SETTING(bool, m_showColliders);
	DEFINE_SETTING(int, m_showCollidersMode);
	DEFINE_SETTING(bool, m_showGrid);
	DEFINE_SETTING(int, m_showGridMode);
	DEFINE_SETTING(bool, m_showGridData);
	DEFINE_SETTING(int, m_showGridDataMode);
	DEFINE_SETTING(bool, m_showParticles);
	DEFINE_SETTING(int, m_showParticlesMode);

	DEFINE_SETTING(glm::vec4, m_selectionColor);
};

#endif