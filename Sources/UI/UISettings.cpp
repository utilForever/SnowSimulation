/*************************************************************************
> File Name: UISettings.cpp
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: UI settings of snow simulation.
> Created Time: 2017/12/28
> Copyright (c) 2017, Chan-Ho Chris Ohk
*************************************************************************/
#include "UISettings.h"

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

    m_gridPosition() = vec3(
        setting.value("gridPositionX", 0.f).toFloat(),
        setting.value("gridPositionY", 0.f).toFloat(),
        setting.value("gridPositionZ", 0.f).toFloat());
    m_gridDimensions() = glm::ivec3(
        setting.value("gridDimensionX", 128).toInt(),
        setting.value("gridDimensionY", 128).toInt(),
        setting.value("gridDimensionZ", 128).toInt());
    m_gridResolution() = setting.value("gridResolution", 0.05f).toFloat();
}