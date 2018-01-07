/*************************************************************************
> File Name: Engine.h
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Engine of snow simulation.
> Created Time: 2018/01/06
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#ifndef SNOW_SIMULATION_ENGINE_H
#define SNOW_SIMULATION_ENGINE_H

#include <Common/Renderable.h>
#include <Geometry/Grid.h>
#include <Simulation/ImplicitCollider.h>
#include <Simulation/Material.h>

#include <QObject>
#include <QTimer>
#include <QVector>

struct cudaGraphicsResource;

struct Node;
struct NodeCache;
struct Particle;
struct ParticleCache;
struct ParticleGrid;
struct ParticleSystem;

struct MitsubaExporter;

class Engine : public QObject, public Renderable
{
    Q_OBJECT

public:
    Engine();
    virtual ~Engine();

    // Returns whether it actually did start
    bool Start(bool exportVolume);
    void Pause();
    void Resume();
    void Stop();
    void Reset();

    float GetSimulationTime();

    void AddParticleSystem(const ParticleSystem& particles);
    ParticleSystem* GetParticleSystem();
    void ClearParticleSystem();

    void SetGrid(const Grid& grid);
    Grid GetGrid();
    void ClearParticleGrid();

    void AddCollider(const ColliderType& t, const Vector3& center, const Vector3& param, const Vector3& velocity);
    QVector<ImplicitCollider>& GetColliders();
    void ClearColliders();

    void InitExporter(QString fPrefix);

    bool IsRunning();

    void Render() override;

    BBox GetBBox(const glm::mat4& ctm) override;
    Vector3 GetCentroid(const glm::mat4& ctm) override;

public slots:
    void Update();

private:
    QTimer m_ticker;

    // CPU data structures
    ParticleSystem* m_particleSystem;
    ParticleGrid* m_particleGrid;
    Grid m_grid;
    QVector<ImplicitCollider> m_colliders;

    // CUDA pointers
    // Particles
    cudaGraphicsResource* m_particlesResource; 
    // Particle grid nodes
    cudaGraphicsResource* m_nodesResource;
    Grid* m_devGrid;

    NodeCache* m_devNodeCaches;

    ParticleCache* m_hostParticleCache;
    ParticleCache* m_devParticleCache;

    ImplicitCollider* m_devColliders;
    Material* m_devMaterial;

    float m_time;

    bool m_busy;
    bool m_running;
    bool m_paused;
    bool m_export;

    MitsubaExporter* m_exporter;

    void InitializeCUDAResources();
    void FreeCUDAResources();
};

#endif