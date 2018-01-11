/*************************************************************************
> File Name: ViewPanel.h
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: View panel of snow simulation.
> Created Time: 2017/12/28
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#ifndef SNOW_SIMULATION_VIEW_PANEL_H
#define SNOW_SIMULATION_VIEW_PANEL_H

class Viewport;
class Scene;

#include <QGLWidget>

class ViewPanel : public QGLWidget
{
	Q_OBJECT

public:
	ViewPanel(QWidget *parent);
	virtual ~ViewPanel();

	bool StartSimulation();
	void StopSimulation();

protected:
	Viewport* m_viewport;

	Scene* m_scene;

	friend class Tool;
	friend class SelectionTool;
	friend class MoveTool;
	friend class RotateTool;
	friend class ScaleTool;
};

#endif