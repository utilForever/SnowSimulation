/*************************************************************************
> File Name: ViewPanel.cpp
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: View panel of snow simulation.
> Created Time: 2017/12/28
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#include "ViewPanel.h"

#include <QWidget>

ViewPanel::ViewPanel(QWidget *parent) :
	QGLWidget(QGLFormat(QGL::SampleBuffers), parent)
{
	
}

ViewPanel::~ViewPanel()
{
	
}

bool ViewPanel::StartSimulation()
{
	return true;
}

void ViewPanel::StopSimulation()
{
	
}