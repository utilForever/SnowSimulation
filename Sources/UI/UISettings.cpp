/*************************************************************************
> File Name: UISettings.cpp
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: UI settings of snow simulation.
> Created Time: 2017/12/28
> Copyright (c) 2018, Chan-Ho Chris Ohk
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

    windowPosition() = setting.value("windowPosition", QPoint(0, 0)).toPoint();
    windowSize() = setting.value("windowSize", QSize(1000, 800)).toSize();

    fillNumParticles() = setting.value("fillNumParticles", 512 * 128).toInt();
    fillResolution() = setting.value("fillResolution", 0.05f).toFloat();
    fillDensity() = setting.value("fillDensity", 150.f).toFloat();

    exportDensity() = setting.value("exportDensity", false).toBool();
    exportVelocity() = setting.value("exportVelocity", false).toBool();
    exportFPS() = setting.value("exportFPS", 24).toInt();
    maxTime() = setting.value("maxTime", 3).toFloat();

    gridPosition() = Vec3(
        setting.value("gridPositionX", 0.f).toFloat(),
        setting.value("gridPositionY", 0.f).toFloat(),
        setting.value("gridPositionZ", 0.f).toFloat());
    gridDimensions() = glm::ivec3(
        setting.value("gridDimensionX", 128).toInt(),
        setting.value("gridDimensionY", 128).toInt(),
        setting.value("gridDimensionZ", 128).toInt());
    gridResolution() = setting.value("gridResolution", 0.05f).toFloat();

    timeStep() = setting.value("timeStep", 1e-5).toFloat();
    implicit() = setting.value("implicit", true).toBool();
    materialPreset() = setting.value("materialPreset", static_cast<int>(SnowMaterialPreset::MAT_DEFAULT)).toInt();

    showContainers() = setting.value("showContainers", true).toBool();
    showContainersMode() = setting.value("showContainersMode", static_cast<int>(MeshMode::WIREFRAME)).toInt();
    showColliders() = setting.value("showColliders", true).toBool();
    showCollidersMode() = setting.value("showCollidersMode", static_cast<int>(MeshMode::SOLID)).toInt();
    showGrid() = setting.value("showGrid", false).toBool();
    showGridMode() = setting.value("showGridMode", static_cast<int>(GridMode::MIN_FACE_CELLS)).toInt();
    showGridData() = setting.value("showGridData", false).toBool();
    showGridDataMode() = setting.value("showGridDataMode", static_cast<int>(GridDataMode::NODE_DENSITY)).toInt();
    showParticles() = setting.value("showParticles", true).toBool();
    showParticlesMode() = setting.value("showParticlesMode", static_cast<int>(ParticlesMode::PARTICLE_MASS)).toInt();

    selectionColor() = glm::vec4(0.302f, 0.773f, 0.839f, 1.f);
}

void UISettings::SaveSettings()
{
    QSettings setting("utilForever", "SnowSimulation");

    setting.setValue("windowPosition", windowPosition());
    setting.setValue("windowSize", windowSize());

    setting.setValue("fillNumParticles", fillNumParticles());
    setting.setValue("fillResolution", fillResolution());
    setting.setValue("fillDensity", fillDensity());

    setting.setValue("exportDensity", exportDensity());
    setting.setValue("exportVelocity", exportVelocity());
    setting.setValue("exportFPS", exportFPS());
    setting.setValue("maxTime", maxTime());

    setting.setValue("gridPositionX", gridPosition().x);
    setting.setValue("gridPositionY", gridPosition().y);
    setting.setValue("gridPositionZ", gridPosition().z);

    setting.setValue("gridDimensionX", gridDimensions().x);
    setting.setValue("gridDimensionY", gridDimensions().y);
    setting.setValue("gridDimensionZ", gridDimensions().z);

    setting.setValue("gridResolution", gridResolution());

    setting.setValue("timeStep", timeStep());
    setting.setValue("implicit", implicit());
    setting.setValue("materialPreset", materialPreset());

    setting.setValue("showContainers", showContainers());
    setting.setValue("showContainersMode", showContainersMode());
    setting.setValue("showColliders", showColliders());
    setting.setValue("showCollidersMode", showCollidersMode());
    setting.setValue("showGrid", showGrid());
    setting.setValue("showGridMode", showGridMode());
    setting.setValue("showGridData", showGridData());
    setting.setValue("showGridDataMode", showGridDataMode());
    setting.setValue("showParticles", showParticles());
    setting.setValue("showParticlesMode", showParticlesMode());
}

Grid UISettings::BuildGrid(const glm::mat4& ctm)
{
    Grid grid;
    const glm::vec4 point = ctm * glm::vec4(0, 0, 0, 1);

    grid.pos = Vec3(point.x, point.y, point.z);
    grid.dim = gridDimensions();
    grid.h = gridResolution();

    return grid;
}