/*************************************************************************
> File Name: Engine.cpp
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Engine of snow simulation.
> Created Time: 2018/01/06
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#include <Common/Util.h>
#include <CUDA/Functions.h>
#include <Geometry/BBox.h>
#include <Simulation/Caches.h>
#include <Simulation/Engine.h>
#include <Simulation/ImplicitCollider.h>
#include <Simulation/Node.h>
#include <Simulation/ParticleGrid.h>
#include <Simulation/ParticleSystem.h>
#include <UI/UISettings.h>

#include <Windows.h>

#include <GL/gl.h>

#include <cuda.h>
#include <cuda_runtime.h>

constexpr int TICKS = 10;

Engine::Engine() :
	m_particleSystem(nullptr), m_particleGrid(nullptr), m_time(0.f),
	m_running(false), m_paused(false), m_exporter(nullptr)
{
	m_particleSystem = new ParticleSystem;
	m_particleGrid = new ParticleGrid;

	m_hostParticleCache = nullptr;

	assert(connect(&m_ticker, SIGNAL(timeout()), this, SLOT(update())));
}

Engine::~Engine()
{
	if (m_running == true)
	{
		Stop();
	}

	if (m_particleSystem != nullptr)
	{
		delete m_particleSystem;
		m_particleSystem = nullptr;
	}

	if (m_particleGrid != nullptr)
	{
		delete m_particleGrid;
		m_particleGrid = nullptr;
	}

	if (m_hostParticleCache != nullptr)
	{
		delete m_hostParticleCache;
		m_hostParticleCache = nullptr;
	}

	if (m_exporter != nullptr)
	{
		delete m_exporter;
		m_exporter = nullptr;
	}
}

bool Engine::Start(bool exportVolume)
{
	if (m_particleSystem->Size() > 0 && !m_grid.IsEmpty() && m_running == false)
	{
		m_export = exportVolume;
		if (m_export == true)
		{
			m_exporter->Reset(m_grid);
		}

		InitializeCUDAResources();
		m_running = true;

		LOG("Simulation started");

		m_ticker.start(TICKS);

		return true;
	}
	
	if (m_particleSystem->Size() == 0)
	{
		LOG("Empty particle system.");
	}

	if (m_grid.IsEmpty())
	{
		LOG("Empty simulation grid.");
	}

	if (m_running)
	{
		LOG("Simulation already running.");
	}

	return false;
}

void Engine::Pause()
{
	m_ticker.stop();
	m_paused = true;
}

void Engine::Resume()
{
	if (m_paused == true)
	{
		m_paused = false;
		
		if (m_running == true)
		{
			m_ticker.start(TICKS);
		}
	}
}

void Engine::Stop()
{
	LOG("Simulation stopped");
	
	m_ticker.stop();
	FreeCUDAResources();
	m_running = false;
}

void Engine::Reset()
{
	if (m_running == false)
	{
		ClearColliders();
		ClearParticleSystem();
		ClearParticleGrid();
		m_time = 0.f;
	}
}

float Engine::GetSimulationTime()
{
	return m_time;
}

void Engine::AddParticleSystem(const ParticleSystem& particles)
{
	QVector<Particle> parts = particles.GetParticles();
	*m_particleSystem += particles;
}

ParticleSystem* Engine::GetParticleSystem()
{
	return m_particleSystem;
}

void Engine::ClearParticleSystem()
{
	m_particleSystem->Clear();
}

void Engine::SetGrid(const Grid& grid)
{
	m_grid = grid;
	m_particleGrid->SetGrid(grid);
}

Grid Engine::GetGrid()
{
	return m_grid;
}

void Engine::ClearParticleGrid()
{
	m_particleGrid->Clear();
}

void Engine::AddCollider(const ImplicitCollider& collider)
{
	m_colliders += collider;
}

void Engine::AddCollider(const ColliderType& t, const Vector3& center, const Vector3& param, const Vector3& velocity)
{
	const ImplicitCollider& col = ImplicitCollider(t, center, param, velocity);
	m_colliders += col;
}

QVector<ImplicitCollider>& Engine::GetColliders()
{
	return m_colliders;
}

void Engine::ClearColliders()
{
	m_colliders.clear();
}

void Engine::InitExporter(QString fPrefix)
{
	m_exporter = new MitsubaExporter(fPrefix, UISettings::exportFPS());
}

bool Engine::IsRunning()
{
	return m_running;
}

void Engine::Render()
{
	if (UISettings::showParticles() == true)
	{
		m_particleSystem->Render();
	}

	if (UISettings::showGridData() == true && m_running == true)
	{
		m_particleGrid->Render();
	}
}

BBox Engine::GetBBox(const glm::mat4& ctm)
{
	return m_particleGrid->GetBBox(ctm);
}

Vector3 Engine::GetCentroid(const glm::mat4& ctm)
{
	return m_particleGrid->GetCentroid(ctm);
}

void Engine::Update()
{
	if (m_busy == false && m_running == true && m_paused == false)
	{
		m_busy = true;

		cudaGraphicsMapResources(1, &m_particlesResource, nullptr);
		
		Particle* devParticles;
		size_t size;
		
		cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&devParticles), &size, m_particlesResource);
		cudaDeviceSynchronize();

		if (static_cast<int>(size / sizeof(Particle)) != m_particleSystem->Size())
		{
			LOG("Particle resource error : %llu bytes (%llu expected)", size, m_particleSystem->Size() * sizeof(Particle));
		}

		cudaGraphicsMapResources(1, &m_nodesResource, nullptr);
		
		Node* devNodes;
		
		cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&devNodes), &size, m_nodesResource);
		cudaDeviceSynchronize();

		if (static_cast<int>(size / sizeof(Node)) != m_particleGrid->GetSize())
		{
			LOG("Grid nodes resource error : %llu bytes (%llu expected)", size, m_particleGrid->GetSize() * sizeof(Node));
		}

		UpdateParticles(
			devParticles, m_devParticleCache, m_hostParticleCache, m_particleSystem->Size(), m_devGrid,
			devNodes, m_devNodeCaches, m_grid.GetNumOfNodes(), m_devColliders, m_colliders.size(),
			UISettings::timeStep(), UISettings::implicit());

		if (m_export && (m_time - m_exporter->GetLastUpdateTime() >= m_exporter->GetSPF()))
		{
			cudaMemcpy(m_exporter->GetNodesPtr(), devNodes, m_grid.GetNumOfNodes() * sizeof(Node), cudaMemcpyDeviceToHost);
			m_exporter->RunExportThread(m_time);
		}

		cudaGraphicsUnmapResources(1, &m_particlesResource, nullptr);
		cudaGraphicsUnmapResources(1, &m_nodesResource, nullptr);
		cudaDeviceSynchronize();

		m_time += UISettings::timeStep();

		// user can adjust max export time dynamically
		if (m_time >= UISettings::maxTime())
		{
			Stop();
			LOG("Simulation completed");
		}

		m_busy = false;
	}
	else
	{
		if (m_running == false)
		{
			LOG("Simulation not running...");
		}

		if (m_paused == true)
		{
			LOG("Simulation paused...");
		}
	}
}

void Engine::InitializeCUDAResources()
{
	LOG("Initializing CUDA resources...");

	// Particles
	RegisterVBO(&m_particlesResource, m_particleSystem->GetVBO());
	float particlesSize = m_particleSystem->Size() * sizeof(Particle) / 1e6;
	LOG("Allocated %.2f MB for particle system.", particlesSize);

	int numNodes = m_grid.GetNumOfNodes();
	int numParticles = m_particleSystem->Size();

	// Grid Nodes
	RegisterVBO(&m_nodesResource, m_particleGrid->GetVBO());
	float nodesSize = numNodes * sizeof(Node) / 1e6;
	LOG("Allocating %.2f MB for grid nodes.", nodesSize);

	// Grid
	cudaMalloc(reinterpret_cast<void**>(&m_devGrid), sizeof(Grid));
	cudaMemcpy(m_devGrid, &m_grid, sizeof(Grid), cudaMemcpyHostToDevice);
	
	// Colliders
	cudaMalloc(reinterpret_cast<void**>(&m_devColliders), m_colliders.size() * sizeof(ImplicitCollider));
	cudaMemcpy(m_devColliders, m_colliders.data(), m_colliders.size() * sizeof(ImplicitCollider), cudaMemcpyHostToDevice);

	// Caches
	cudaMalloc(reinterpret_cast<void**>(&m_devNodeCaches), numNodes * sizeof(NodeCache));
	cudaMemset(m_devNodeCaches, 0, numNodes * sizeof(NodeCache));
	float nodeCachesSize = numNodes * sizeof(NodeCache) / 1e6;
	LOG("Allocating %.2f MB for implicit update node cache.", nodeCachesSize);

	if (m_hostParticleCache != nullptr)
	{
		delete m_hostParticleCache;
		m_hostParticleCache = nullptr;
	}

	m_hostParticleCache = new ParticleCache;
	cudaMalloc(reinterpret_cast<void**>(&m_hostParticleCache->sigmas), numParticles * sizeof(Matrix3));
	cudaMalloc(reinterpret_cast<void**>(&m_hostParticleCache->Aps), numParticles * sizeof(Matrix3));
	cudaMalloc(reinterpret_cast<void**>(&m_hostParticleCache->FeHats), numParticles * sizeof(Matrix3));
	cudaMalloc(reinterpret_cast<void**>(&m_hostParticleCache->ReHats), numParticles * sizeof(Matrix3));
	cudaMalloc(reinterpret_cast<void**>(&m_hostParticleCache->SeHats), numParticles * sizeof(Matrix3));
	cudaMalloc(reinterpret_cast<void**>(&m_hostParticleCache->dFs), numParticles * sizeof(Matrix3));
	cudaMalloc(reinterpret_cast<void**>(&m_devParticleCache), sizeof(ParticleCache));
	cudaMemcpy(m_devParticleCache, m_hostParticleCache, sizeof(ParticleCache), cudaMemcpyHostToDevice);
	float particleCachesSize = numParticles * 6 * sizeof(Matrix3) / 1e6;
	LOG("Allocating %.2f MB for implicit update particle caches.", particleCachesSize);

	LOG("Allocated %.2f MB in total", particlesSize + nodesSize + nodeCachesSize + particleCachesSize);

	LOG("Computing particle volumes...");
	cudaGraphicsMapResources(1, &m_particlesResource, nullptr);
	Particle* devParticles;
	size_t size;
	cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&devParticles), &size, m_particlesResource);
	if (static_cast<int>(size / sizeof(Particle)) != m_particleSystem->Size())
	{
		LOG("Particle resource error : %llu bytes (%llu expected)", size, m_particleSystem->Size() * sizeof(Particle));
	}
	InitializeParticleVolumes(devParticles, m_particleSystem->Size(), m_devGrid, numNodes);
	cudaGraphicsUnmapResources(1, &m_particlesResource, nullptr);

	LOG("Initialization complete.");
}

void Engine::FreeCUDAResources()
{
	LOG("Freeing CUDA resources...");
	
	UnregisterVBO(m_particlesResource);
	UnregisterVBO(m_nodesResource);
	
	cudaFree(m_devGrid);
	cudaFree(m_devColliders);
	cudaFree(m_devNodeCaches);

	// Free the particle cache using the host structure
	cudaFree(m_hostParticleCache->sigmas);
	cudaFree(m_hostParticleCache->Aps);
	cudaFree(m_hostParticleCache->FeHats);
	cudaFree(m_hostParticleCache->ReHats);
	cudaFree(m_hostParticleCache->SeHats);
	cudaFree(m_hostParticleCache->dFs);
	
	if (m_hostParticleCache != nullptr)
	{
		delete m_hostParticleCache;
		m_hostParticleCache = nullptr;
	}

	cudaFree(m_devParticleCache);

	cudaFree(m_devMaterial);
}