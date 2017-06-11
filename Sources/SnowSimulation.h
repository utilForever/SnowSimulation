#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_SnowSimulation.h"

class SnowSimulation : public QMainWindow
{
	Q_OBJECT

public:
	SnowSimulation(QWidget *parent = Q_NULLPTR);

private:
	Ui::SnowSimulationClass ui;
};
