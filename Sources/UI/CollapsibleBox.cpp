/*************************************************************************
> File Name: CollapsibleBox.cpp
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Collapsible box of snow simulation.
> Created Time: 2018/01/09
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#include <UI/CollapsibleBox.h>
#include <UI/UISettings.h>

#include <QApplication>
#include <QLayout>
#include <QMouseEvent>
#include <QPalette>


CollapsibleBox::CollapsibleBox(QWidget* widget) :
	QGroupBox(widget), m_rawTitle(""), m_clicked(false), m_collapsed(false)
{
	this->setAutoFillBackground(true);
}

CollapsibleBox::~CollapsibleBox()
{
	UISettings::SetSetting(this->objectName() + "IsCollapsed", m_collapsed);
}

bool CollapsibleBox::IsCollapsed() const
{
	return m_collapsed;
}

void CollapsibleBox::Init()
{
	SetCollapsed(UISettings::GetSetting(this->objectName() + "IsCollapsed", false).toBool());
}

void CollapsibleBox::mousePressEvent(QMouseEvent *event)
{
	if (!childrenRect().contains(event->pos()))
	{
		QPalette palette = this->palette();
		palette.setColor(QPalette::Window, palette.dark().color());
		palette.setColor(QPalette::Button, palette.dark().color());
		
		SetWidgetPalette(this, palette);
		m_clicked = true;
	}
}

void CollapsibleBox::mouseReleaseEvent(QMouseEvent*)
{
	if (m_clicked == true)
	{
		SetCollapsed(!m_collapsed);
		SetWidgetPalette(this, QApplication::palette());   
		m_clicked = false;
	}
}

void CollapsibleBox::SetCollapsed(bool collapsed)
{
	m_collapsed = collapsed;
	
	for (int i = 0; i < children().size(); ++i)
	{
		children()[i]->setProperty("visible", !collapsed);
	}
	
	if (m_collapsed == true)
	{
		setMaximumHeight(1.25 * fontMetrics().height());
	}
	else
	{
		setMaximumHeight(16777215);
	}
	
	this->setTitle(m_rawTitle);
}

void CollapsibleBox::SetWidgetPalette(QWidget* widget, const QPalette& palette)
{
	widget->setPalette(palette);
	
	QList<QWidget*> children = widget->findChildren<QWidget*>();
	for (int i = 0; i < children.size(); ++i)
	{
		SetWidgetPalette(children[i], palette);
	}
}