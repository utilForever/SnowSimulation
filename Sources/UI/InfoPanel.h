/*************************************************************************
> File Name: InfoPanel.h
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Info panel of snow simulation.
> Created Time: 2018/01/10
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#ifndef SNOW_SIMULATION_INFO_PANEL_H
#define SNOW_SIMULATION_INFO_PANEL_H

#include <Common/Renderable.h>

#ifndef GLM_FORCE_RADIANS
#define GLM_FORCE_RADIANS
#endif

#include <glm/vec2.hpp>

#include <QFont>
#include <QHash>
#include <QString>


class ViewPanel;

class InfoPanel
{
    struct Entry
    {
        QString key;
        QString value;
        glm::ivec2 pos;
        
        Entry()
        {
            // Do nothing
        }

        Entry(const QString& k, const QString& v) : key(k), value(v)
        {
            // Do nothing
        }

        Entry(const Entry& entry) : key(entry.key), value(entry.value), pos(entry.pos)
        {
            // Do nothing
        }
    };

public:
    InfoPanel(ViewPanel* panel);
    virtual ~InfoPanel();

    void SetFont(const QFont& font);

    void AddInfo(const QString& key, const QString& value = QString());
    void SetInfo(const QString& key, const QString& value, bool layout = true);
    void RemoveInfo(const QString& key);

    void Render();

private:
    void UpdateLayout();

    ViewPanel* m_panel;
    QHash<QString, Entry> m_info;
    QList<QString> m_order;

    QFont m_font;
    int m_spacing, m_margin;
};

#endif