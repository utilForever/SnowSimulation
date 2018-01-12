/*************************************************************************
> File Name: SceneIO.h
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Scene IO of snow simulation.
> Created Time: 2018/01/08
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#ifndef SNOW_SIMULATION_SCENE_IO_H
#define SNOW_SIMULATION_SCENE_IO_H

#include <glm/mat4x4.hpp>

#include <QtXml/QtXml>

#include <iostream>

struct Vector3;

struct SimulationParameters;
struct ImplicitCollider;

class Scene;
class Engine;
class Grid;
class ParticleSystem;

class SceneIO
{
public:
	SceneIO();

	bool Read(QString fileName, Scene* scene, Engine* engine);
	void Write(Scene* scene, Engine* engine);

	QString GetSceneFile();
	void SetSceneFile(QString fileName);

private:
	// import functions
	void ApplySimulationParameters();
	void ApplyExportSettings();
	void ApplyParticleSystem(Scene* scene);
	void ApplyGrid(Scene* scene);
	void ApplyColliders(Scene* scene, Engine* engine);

	// export functions

	void AppendSimulationParameters(QDomElement root, float timeStep);
	void AppendParticleSystem(QDomElement root, Scene* scene);
	void AppendGrid(QDomElement root, Scene* scene);
	void AppendColliders(QDomElement root, Scene* scene);
	void AppendExportSettings(QDomElement root);

	// low level DOM node helpers
	void AppendString(QDomElement node, const QString name, const QString value);
	void AppendInt(QDomElement node, const QString name, const int i);
	void AppendFloat(QDomElement node, const QString name, const float f);
	void AppendVector(QDomElement node, const QString name, const Vector3 v);
	void AppendDim(QDomElement node, const QString name, const glm::ivec3 iv);
	void AppendMatrix(QDomElement node, const QString name, const glm::mat4 mat);

	// i.e. /users/jcarberr/Castle
	QString m_sceneFilePrefix;
	// XML document
	QDomDocument m_document;
};

#endif