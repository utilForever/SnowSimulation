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

	enum class ParticlesMode
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

	DEFINE_SETTING(QPoint, windowPosition);
	DEFINE_SETTING(QSize, windowSize);

	// Filling
	DEFINE_SETTING(int, fillNumParticles);
	DEFINE_SETTING(float, fillDensity);
	DEFINE_SETTING(float, fillResolution);

	// Exporting
	DEFINE_SETTING(bool, exportDensity);
	DEFINE_SETTING(bool, exportVelocity);
	DEFINE_SETTING(int, exportFPS);
	DEFINE_SETTING(float, maxTime);

	DEFINE_SETTING(Vec3, gridPosition);
	DEFINE_SETTING(glm::ivec3, gridDimensions);
	DEFINE_SETTING(float, gridResolution);

	DEFINE_SETTING(float, timeStep);
	DEFINE_SETTING(bool, implicit);
	DEFINE_SETTING(int, materialPreset);

	DEFINE_SETTING(bool, showContainers);
	DEFINE_SETTING(int, showContainersMode);
	DEFINE_SETTING(bool, showColliders);
	DEFINE_SETTING(int, showCollidersMode);
	DEFINE_SETTING(bool, showGrid);
	DEFINE_SETTING(int, showGridMode);
	DEFINE_SETTING(bool, showGridData);
	DEFINE_SETTING(int, showGridDataMode);
	DEFINE_SETTING(bool, showParticles);
	DEFINE_SETTING(int, showParticlesMode);

	DEFINE_SETTING(glm::vec4, selectionColor);
};

#endif