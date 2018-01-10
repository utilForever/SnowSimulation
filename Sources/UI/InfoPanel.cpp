/*************************************************************************
> File Name: InfoPanel.cpp
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Info panel of snow simulation.
> Created Time: 2018/01/10
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#include <UI/InfoPanel.h>
#include <UI/ViewPanel.h>

#include <QFontMetrics>
#include <QGridLayout>
#include <QLabel>

InfoPanel::InfoPanel(ViewPanel* panel) :
	m_panel(panel), m_font(QFont("Helvetica", 10)), m_spacing(2), m_margin(6)
{
	// Do nothing
}

InfoPanel::~InfoPanel()
{
	// Do nothing
}

void InfoPanel::SetFont(const QFont& font)
{
	m_font = font;
}

void InfoPanel::AddInfo(const QString& key, const QString& value)
{
	m_info.insert(key, Entry(key, value));
	m_order += key;
	UpdateLayout();
}

void InfoPanel::SetInfo(const QString& key, const QString& value, bool layout)
{
	if (!m_info.contains(key))
	{
		m_info.insert(key, Entry(key, value));
		m_order += key;
	}
	else
	{
		m_info[key].value = value;
	}
	
	if (layout == true)
	{
		UpdateLayout();
	}
}

void InfoPanel::RemoveInfo(const QString& key)
{
	if (m_info.contains(key))
	{
		m_info.remove(key);
		m_order.removeAll(key);
	}

	UpdateLayout();
}

void InfoPanel::Render()
{
	glPushAttrib(GL_COLOR_BUFFER_BIT);
	glColor4f(1.f, 1.f, 1.f, 1.f);
	
	for (QHash<QString, Entry>::const_iterator iter = m_info.begin(); iter != m_info.end(); ++iter)
	{
		const Entry& entry = iter.value();
		m_panel->renderText(entry.pos.x, entry.pos.y, QString("%1: %2").arg(entry.key, entry.value));
	}

	glPopAttrib();
}

void InfoPanel::UpdateLayout()
{
	QFontMetrics metrics(m_font);
	int h = metrics.height();
	int y = metrics.ascent() + m_margin;
	int x0 = m_margin;

	for (int i = 0; i < m_order.size(); ++i)
	{
		Entry& entry = m_info[m_order[i]];
		entry.pos.y = y;
		entry.pos.x = x0;
		y += (h + m_spacing);
	}
}