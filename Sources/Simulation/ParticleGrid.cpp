/*************************************************************************
> File Name: ParticleGrid.cpp
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: ParticleGrid structure of snow simulation.
> Created Time: 2018/01/02
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#include <Common/Util.h>
#include <Geometry/BBox.h>
#include <Simulation/ParticleGrid.h>
#include <UI/UISettings.h>

#include <GL/glew.h>
#include <GL/gl.h>

#include <QGLShaderProgram>
#include <QVector>

ParticleGrid::ParticleGrid() :
    m_size(0), m_glIndices(0), m_glVBO(0), m_glVAO(0)
{
    // Do nothing
}

ParticleGrid::~ParticleGrid()
{
    DeleteBuffers();
}

void ParticleGrid::Render()
{
    if (GetSize() > 0)
    {
        if (!HasBuffers())
        {
            BuildBuffers();
        }

        QGLShaderProgram* shader = ParticleGrid::GetShader();
        if (shader != nullptr)
        {
            glPushAttrib(GL_VERTEX_PROGRAM_POINT_SIZE);
            glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
            shader->bind();
            shader->setUniformValue("pos", m_grid.pos.x, m_grid.pos.y, m_grid.pos.z);
            shader->setUniformValue("dim", static_cast<float>(m_grid.dim.x), static_cast<float>(m_grid.dim.y), static_cast<float>(m_grid.dim.z));
            shader->setUniformValue("h", m_grid.h);
            shader->setUniformValue("density", UISettings::fillDensity());
            shader->setUniformValue("mode", UISettings::showGridDataMode());
        }
        else
        {
            glPushAttrib(GL_LIGHTING_BIT);
            glDisable(GL_LIGHTING);
            glColor4f(1.0f, 1.0f, 1.0f, 0.f);
            glPointSize(1.f);
        }

        glPushAttrib(GL_DEPTH_BUFFER_BIT);
        glDepthMask(false);

        glPushAttrib(GL_COLOR_BUFFER_BIT);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_POINT_SMOOTH);
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);

        glEnable(GL_ALPHA_TEST);
        glAlphaFunc(GL_GREATER, 0.05f);

        glBindVertexArray(m_glVAO);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_glIndices);
        glDrawElements(GL_POINTS, m_size, GL_UNSIGNED_INT, static_cast<void*>(nullptr));
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        glBindVertexArray(0);

        glPopAttrib();
        glPopAttrib();

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
}

void ParticleGrid::Clear()
{
    DeleteBuffers();
}

void ParticleGrid::SetGrid(const Grid& grid)
{
    m_grid = grid;
    m_size = m_grid.GetNumOfNodes();
    DeleteBuffers();
}

Grid ParticleGrid::GetGrid() const
{
    return m_grid;
}

GLuint ParticleGrid::GetVBO()
{
    if (!HasBuffers())
    {
        BuildBuffers();
    }

    return m_glVBO;
}

int ParticleGrid::GetSize() const
{
    return m_size;
}

int ParticleGrid::GetNodeCount() const
{
    return m_size;
}

BBox ParticleGrid::GetBBox(const glm::mat4& ctm)
{
    return BBox(m_grid).GetBBox(ctm);
}

Vector3 ParticleGrid::GetCentroid(const glm::mat4& ctm)
{
    return BBox(m_grid).GetCentroid(ctm);
}

bool ParticleGrid::HasBuffers() const
{
    return m_glVBO > 0 && glIsBuffer(m_glVBO);
}

void ParticleGrid::BuildBuffers()
{
    DeleteBuffers();

    Node* data = new Node[m_size];
    memset(data, 0, m_size * sizeof(Node));

    // Build VBO
    glGenBuffers(1, &m_glVBO);
    glBindBuffer(GL_ARRAY_BUFFER, m_glVBO);
    glBufferData(GL_ARRAY_BUFFER, m_size * sizeof(Node), data, GL_DYNAMIC_DRAW);

    delete[] data;

    // Build VAO
    glGenVertexArrays(1, &m_glVAO);
    glBindVertexArray(m_glVAO);

    // Mass attribute
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, sizeof(Node), static_cast<void*>(nullptr));

    // Velocity attribute
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Node), reinterpret_cast<void*>(sizeof(GLfloat)));

    // Force attribute
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(Node), reinterpret_cast<void*>(sizeof(GLfloat) + 2 * sizeof(Vector3)));

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Indices (needed to access vertex index in shader)
    QVector<unsigned int> indices;
    for (unsigned int i = 0; i < static_cast<unsigned int>(m_size); ++i)
    {
        indices += i;
    }

    glGenBuffers(1, &m_glIndices);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_glIndices);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void ParticleGrid::DeleteBuffers()
{
    // Delete OpenGL VBO and unregister with CUDA
    if (HasBuffers())
    {
        glBindBuffer(GL_ARRAY_BUFFER, m_glVBO);
        glDeleteBuffers(1, &m_glVBO);
        glDeleteVertexArrays(1, &m_glVAO);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_glIndices);
        glDeleteBuffers(1, &m_glIndices);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    }

    m_glVBO = 0;
    m_glVAO = 0;
    m_glIndices = 0;
}

QGLShaderProgram* ParticleGrid::m_shader = nullptr;

QGLShaderProgram* ParticleGrid::GetShader()
{
    if (m_shader == nullptr)
    {
        const QGLContext* context = QGLContext::currentContext();
        m_shader = new QGLShaderProgram(context);
        
        if (!m_shader->addShaderFromSourceFile(QGLShader::Vertex, ":/shaders/particlegrid.vert") ||
            !m_shader->addShaderFromSourceFile(QGLShader::Fragment, ":/shaders/particlegrid.frag"))
        {
            LOG("ParticleGrid::shader() : Compile error: \n%s\n", STR(m_shader->log().trimmed()));
            
            if (m_shader != nullptr)
            {
                delete m_shader;
                m_shader = nullptr;
            }
        }
        else
        {
            m_shader->bindAttributeLocation("nodeMass", 0);
            m_shader->bindAttributeLocation("nodeVelocity", 1);
            m_shader->bindAttributeLocation("nodeForce", 2);
            glBindFragDataLocation(m_shader->programId(), 0, "fragmentColor");
            
            if (!m_shader->link())
            {
                LOG("ParticleGrid::shader() : Link error: \n%s\n", STR(m_shader->log().trimmed()));
                
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