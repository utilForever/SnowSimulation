#include "ViewPanel.h"

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