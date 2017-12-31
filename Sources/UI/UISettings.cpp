/*************************************************************************
> File Name: UISettings.cpp
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: UI settings of snow simulation.
> Created Time: 2017/12/28
> Copyright (c) 2017, Chan-Ho Chris Ohk
*************************************************************************/
#include <Geometry/Grid.h>
#include <UI/UISettings.h>

#include <QSettings>

UISettings* UISettings::m_instance = nullptr;

UISettings* UISettings::GetInstance()
{
    if (m_instance == nullptr)
    {
        m_instance = new UISettings();
    }

    return m_instance;
}

void UISettings::DeleteInstance()
{
    if (m_instance != nullptr)
    {
        delete m_instance;
        m_instance = nullptr;
    }
}

QVariant UISettings::GetSetting(const QString& name, const QVariant& value)
{
    QSettings setting("utilForever", "SnowSimulation");
    return setting.value(name, value);
}

void UISettings::SetSetting(const QString& name, const QVariant& value)
{
    QSettings setting("utilForever", "SnowSimulation");
    setting.setValue(name, value);
}

void UISettings::LoadSettings()
{
    QSettings setting("utilForever", "SnowSimulation");

    m_windowPosition() = setting.value("windowPosition", QPoint(0, 0)).toPoint();
    m_windowSize() = setting.value("windowSize", QSize(1000, 800)).toSize();

    m_fillNumParticles() = setting.value("fillNumParticles", 512 * 128).toInt();
    m_fillResolution() = setting.value("fillResolution", 0.05f).toFloat();
    m_fillDensity() = setting.value("fillDensity", 150.f).toFloat();

    m_exportDensity() = setting.value("exportDensity", false).toBool();
    m_exportVelocity() = setting.value("exportVelocity", false).toBool();
    m_exportFPS() = setting.value("exportFPS", 24).toInt();
    m_maxTime() = setting.value("maxTime", 3).toFloat();

    m_gridPosition() = Vec3(
        setting.value("gridPositionX", 0.f).toFloat(),
        setting.value("gridPositionY", 0.f).toFloat(),
        setting.value("gridPositionZ", 0.f).toFloat());
    m_gridDimensions() = glm::ivec3(
        setting.value("gridDimensionX", 128).toInt(),
        setting.value("gridDimensionY", 128).toInt(),
        setting.value("gridDimensionZ", 128).toInt());
    m_gridResolution() = setting.value("gridResolution", 0.05f).toFloat();

    m_timeStep() = setting.value("timeStep", 1e-5).toFloat();
    m_implicit() = setting.value("implicit", true).toBool();
    m_materialPreset() = setting.value("materialPreset", static_cast<int>(SnowMaterialPreset::MAT_DEFAULT)).toInt();

    m_showContainers() = setting.value("showContainers", true).toBool();
    m_showContainersMode() = setting.value("showContainersMode", static_cast<int>(MeshMode::WIREFRAME)).toInt();
    m_showColliders() = setting.value("showColliders", true).toBool();
    m_showCollidersMode() = setting.value("showCollidersMode", static_cast<int>(MeshMode::SOLID)).toInt();
    m_showGrid() = setting.value("showGrid", false).toBool();
    m_showGridMode() = setting.value("showGridMode", static_cast<int>(GridMode::MIN_FACE_CELLS)).toInt();
    m_showGridData() = setting.value("showGridData", false).toBool();
    m_showGridDataMode() = setting.value("showGridDataMode", static_cast<int>(GridDataMode::NODE_DENSITY)).toInt();
    m_showParticles() = setting.value("showParticles", true).toBool();
    m_showParticlesMode() = setting.value("showParticlesMode", static_cast<int>(ParticlesMode::PARTICLE_MASS)).toInt();

    m_selectionColor() = glm::vec4(0.302f, 0.773f, 0.839f, 1.f);
}

void UISettings::SaveSettings()
{
    QSettings setting("utilForever", "SnowSimulation");

    setting.setValue("windowPosition", m_windowPosition());
    setting.setValue("windowSize", m_windowSize());

    setting.setValue("fillNumParticles", m_fillNumParticles());
    setting.setValue("fillResolution", m_fillResolution());
    setting.setValue("fillDensity", m_fillDensity());

    setting.setValue("exportDensity", m_exportDensity());
    setting.setValue("exportVelocity", m_exportVelocity());
    setting.setValue("exportFPS", m_exportFPS());
    setting.setValue("maxTime", m_maxTime());

    setting.setValue("gridPositionX", m_gridPosition().x);
    setting.setValue("gridPositionY", m_gridPosition().y);
    setting.setValue("gridPositionZ", m_gridPosition().z);

    setting.setValue("gridDimensionX", m_gridDimensions().x);
    setting.setValue("gridDimensionY", m_gridDimensions().y);
    setting.setValue("gridDimensionZ", m_gridDimensions().z);

    setting.setValue("gridResolution", m_gridResolution());

    setting.setValue("timeStep", m_timeStep());
    setting.setValue("implicit", m_implicit());
    setting.setValue("materialPreset", m_materialPreset());

    setting.setValue("showContainers", m_showContainers());
    setting.setValue("showContainersMode", m_showContainersMode());
    setting.setValue("showColliders", m_showColliders());
    setting.setValue("showCollidersMode", m_showCollidersMode());
    setting.setValue("showGrid", m_showGrid());
    setting.setValue("showGridMode", m_showGridMode());
    setting.setValue("showGridData", m_showGridData());
    setting.setValue("showGridDataMode", m_showGridDataMode());
    setting.setValue("showParticles", m_showParticles());
    setting.setValue("showParticlesMode", m_showParticlesMode());
}

Grid UISettings::BuildGrid(const glm::mat4& ctm)
{
    Grid grid;
    const glm::vec4 point = ctm * glm::vec4(0, 0, 0, 1);

    grid.pos = Vec3(point.x, point.y, point.z);
    grid.dim = m_gridDimensions();
    grid.h = m_gridResolution();

    return grid;
}