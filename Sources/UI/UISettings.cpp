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