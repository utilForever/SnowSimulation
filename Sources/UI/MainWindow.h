/*************************************************************************
> File Name: MainWindow.h
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Main window UI of snow simulation.
> Created Time: 2017/06/11
> Copyright (c) 2017, Chan-Ho Chris Ohk
*************************************************************************/
#ifndef SNOW_SIMULATION_MAIN_WINDOW_H
#define SNOW_SIMULATION_MAIN_WINDOW_H

#include <QtWidgets/QMainWindow>

// Forward declaration
namespace Ui
{
	class MainWindow;
}

class MainWindow : public QMainWindow
{
	Q_OBJECT

public:
	explicit MainWindow(QWidget* parent = nullptr);
	~MainWindow();

	void StartSimulation();
	void StopSimulation();

private:
	void SetupUI();

	Ui::MainWindow* m_ui;
};

#endif