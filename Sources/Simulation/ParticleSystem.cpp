/*************************************************************************
> File Name: ParticleSystem.cpp
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Particle system of snow simulation.
> Created Time: 2018/01/04
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#include <Common/Util.h>
#include <Geometry/BBox.h>
#include <Simulation/ParticleSystem.h>
#include <UI/UISettings.h>

#include <Windows.h>

#include <GL/glew.h>
#include <GL/gl.h>

#include <QGLShaderProgram>

ParticleSystem::ParticleSystem() : m_glVBO(0), m_glVAO(0)
{
	// Do nothing
}

ParticleSystem::~ParticleSystem()
{
	DeleteBuffers();
}

void ParticleSystem::Clear()
{
	m_particles.clear();
	DeleteBuffers();
}

int ParticleSystem::Size() const
{
	return m_particles.size();
}

void ParticleSystem::Resize(int n)
{
	m_particles.resize(n);
}

Particle* ParticleSystem::GetData()
{
	return m_particles.data();
}

const QVector<Particle>& ParticleSystem::GetParticles() const
{
	return m_particles;
}

QVector<Particle>& ParticleSystem::GetParticles()
{
	return m_particles;
}

void ParticleSystem::Render()
{
	if (!HasBuffers())
	{
		BuildBuffers();
	}

	QGLShaderProgram* shader = ParticleSystem::GetShader();
	if (shader != nullptr)
	{
		glPushAttrib(GL_VERTEX_PROGRAM_POINT_SIZE);
		glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
		shader->bind();
		shader->setUniformValue("mode", UISettings::showParticlesMode());
	}
	else
	{
		glPushAttrib(GL_LIGHTING_BIT);
		glDisable(GL_LIGHTING);
		glColor3f(1.0f, 1.0f, 1.0f);
		glPointSize(1.f);
	}

	glEnable(GL_POINT_SMOOTH);
	glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);

	glBindVertexArray(m_glVAO);
	glDrawArrays(GL_POINTS, 0, m_particles.size());
	glBindVertexArray(0);

	if (shader != nullptr)
	{
		shader->release();
		glPopAttrib();
	}
	else
	{
		glPopAttrib();
	}
}

BBox ParticleSystem::GetBBox(const glm::mat4& ctm)
{
	BBox box;

	for (size_t i = 0; i < m_particles.size(); ++i)
	{
		const Vector3& p = m_particles[i].position;
		glm::vec4 point = ctm * glm::vec4(glm::vec3(p), 1.f);
		box += Vector3(point.x, point.y, point.z);
	}

	return box;
}

Vector3 ParticleSystem::GetCentroid(const glm::mat4& ctm)
{
	Vector3 c(0, 0, 0);

	for (size_t i = 0; i < m_particles.size(); ++i)
	{
		const Vector3 p = m_particles[i].position;
		glm::vec4 point = ctm * glm::vec4(glm::vec3(p), 1.f);
		c += Vector3(point.x, point.y, point.z);
	}

	return c / static_cast<float>(m_particles.size());
}

GLuint ParticleSystem::GetVBO()
{
	if (!HasBuffers())
	{
		BuildBuffers();
	}

	return m_glVBO;
}

void ParticleSystem::Merge(const ParticleSystem& particles)
{
	m_particles += particles.m_particles;
	DeleteBuffers();
}

ParticleSystem& ParticleSystem::operator+=(const ParticleSystem& particles)
{
	m_particles += particles.m_particles;
	DeleteBuffers();
	
	return *this;
}

ParticleSystem& ParticleSystem::operator+=(const Particle& particle)
{
	m_particles.append(particle);
	DeleteBuffers();
	
	return *this;
}

bool ParticleSystem::HasBuffers() const
{
	return m_glVBO > 0 && glIsBuffer(m_glVBO);
}

void ParticleSystem::BuildBuffers()
{
	DeleteBuffers();

	// Build OpenGL VBO
	glGenBuffers(1, &m_glVBO);
	glBindBuffer(GL_ARRAY_BUFFER, m_glVBO);
	glBufferData(GL_ARRAY_BUFFER, m_particles.size() * sizeof(Particle), m_particles.data(), GL_DYNAMIC_DRAW);

	// Build OpenGL VAO
	glGenVertexArrays(1, &m_glVAO);
	glBindVertexArray(m_glVAO);

	// offset within particle struct
	std::size_t offset = 0; 

	// Position attribute
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Particle), reinterpret_cast<void*>(offset));
	offset += sizeof(Vector3);

	// Velocity attribute
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Particle), reinterpret_cast<void*>(offset));
	offset += sizeof(Vector3);

	// Mass attribute
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(Particle), reinterpret_cast<void*>(offset));
	offset += sizeof(GLfloat);

	// Volume attribute
	glEnableVertexAttribArray(3);
	glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, sizeof(Particle), reinterpret_cast<void*>(offset));
	offset += sizeof(GLfloat);
	offset += 2 * sizeof(Matrix3);

	// lambda (stiffness) attribute
	// skip to material.xi
	offset += 2 * sizeof(GLfloat);
	glEnableVertexAttribArray(4);
	glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, sizeof(Particle), reinterpret_cast<void*>(offset));
	offset += sizeof(GLfloat);

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void ParticleSystem::DeleteBuffers()
{
	// Delete OpenGL VBO and unregister with CUDA
	if (HasBuffers())
	{
		glBindBuffer(GL_ARRAY_BUFFER, m_glVBO);
		glDeleteBuffers(1, &m_glVBO);
		glDeleteVertexArrays(1, &m_glVAO);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	m_glVBO = 0;
	m_glVAO = 0;
}

void ParticleSystem::SetVelocity()
{
	for (int i = 0; i < m_particles.size(); i++)
	{
		m_particles[i].velocity = m_velocityVector * m_VelocityMagnitude;
	}
}

QGLShaderProgram* ParticleSystem::m_shader = nullptr;

QGLShaderProgram* ParticleSystem::GetShader()
{
	if (m_shader == nullptr)
	{
		const QGLContext* context = QGLContext::currentContext();
		m_shader = new QGLShaderProgram(context);
		
		if (!m_shader->addShaderFromSourceFile(QGLShader::Vertex, ":/shaders/particlesystem.vert") ||
			!m_shader->addShaderFromSourceFile(QGLShader::Fragment, ":/shaders/particlesystem.frag"))
		{
			LOG("ParticleSystem::shader() : Compile error: \n%s\n", STR(m_shader->log().trimmed()));
			
			if (m_shader != nullptr)
			{
				delete m_shader;
				m_shader = nullptr;
			}
		}
		else
		{
			m_shader->bindAttributeLocation("particlePosition", 0);
			m_shader->bindAttributeLocation("particleVelocity", 1);
			m_shader->bindAttributeLocation("particleMass", 2);
			m_shader->bindAttributeLocation("particleVolume", 3);
			m_shader->bindAttributeLocation("particleStiffness", 4);
			glBindFragDataLocation(m_shader->programId(), 0, "fragmentColor");
			
			if (!m_shader->link())
			{
				LOG("ParticleSystem::shader() : Link error: \n%s\n", STR(m_shader->log().trimmed()));
				
				if (m_shader != nullptr)
				{
					delete m_shader;
					m_shader = nullptr;
				}
			}
		}
	}

	return m_shader;
}