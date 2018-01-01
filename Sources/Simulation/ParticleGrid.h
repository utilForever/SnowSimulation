/*************************************************************************
> File Name: ParticleGrid.h
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: ParticleGrid structure of snow simulation.
> Created Time: 2018/01/02
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#ifndef SNOW_SIMULATION_PARTICLE_GRID_H
#define SNOW_SIMULATION_PARTICLE_GRID_H

#include <Common/Renderable.h>
#include <Geometry/Grid.h>
#include <Simulation/Node.h>
#include <Simulation/Particle.h>

class QGLShaderProgram;

using GLuint = unsigned int;

class ParticleGrid : public Renderable
{
public:
    ParticleGrid();
    virtual ~ParticleGrid();

    void Render() override;

    void Clear();

    void SetGrid(const Grid& grid);
    Grid GetGrid() const;

    GLuint GetVBO();

    int GetSize() const;
    int GetNodeCount() const;

    virtual BBox GetBBox(const glm::mat4& ctm);
    virtual Vector3 GetCentroid(const glm::mat4& ctm);

    bool HasBuffers() const;
    void BuildBuffers();
    void DeleteBuffers();

protected:
    static QGLShaderProgram* GetShader();

    static QGLShaderProgram* m_shader;

    Grid m_grid;
    int m_size;
    GLuint m_glIndices;
    GLuint m_glVBO;
    GLuint m_glVAO;
};

#endif