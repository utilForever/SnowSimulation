/*************************************************************************
> File Name: ParticleSystem.h
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Particle system of snow simulation.
> Created Time: 2018/01/04
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#ifndef SNOW_SIMULATION_PARTICLE_SYSTEM_H
#define SNOW_SIMULATION_PARTICLE_SYSTEM_H

#include <Common/Renderable.h>
#include <Geometry/Grid.h>
#include <Simulation/Particle.h>

#include <QVector>

class QGLShaderProgram;

using GLuint = unsigned int;

class ParticleSystem : public Renderable
{
public:
	ParticleSystem();
	virtual ~ParticleSystem();

	void Clear();
	int Size() const;
	void Resize(int n);

	Particle* GetData();
	const QVector<Particle>& GetParticles() const;
	QVector<Particle>& GetParticles();

    void Render() override;

    BBox GetBBox(const glm::mat4& ctm) override;
    Vector3 GetCentroid(const glm::mat4& ctm) override;

	GLuint GetVBO();

	void Merge(const ParticleSystem& particles);

	ParticleSystem& operator+=(const ParticleSystem& particles);
	ParticleSystem& operator+=(const Particle& particle);

	bool HasBuffers() const;
	void BuildBuffers();
	void DeleteBuffers();
	void SetVelocity();

protected:
	static QGLShaderProgram* m_shader;
	static QGLShaderProgram* GetShader();

	QVector<Particle> m_particles;
	GLuint m_glVBO;
	GLuint m_glVAO;
};

#endif