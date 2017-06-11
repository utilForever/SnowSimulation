#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_SnowSimulation.h"

class MainWindow : public QMainWindow
{
	Q_OBJECT

public:
	MainWindow(QWidget *parent = Q_NULLPTR);

private:
	Ui::SnowSimulationClass ui;
};
