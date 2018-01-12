/*************************************************************************
> File Name: MainWindow.h
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Main window UI of snow simulation.
> Created Time: 2017/06/11
> Copyright (c) 2018, Chan-Ho Chris Ohk
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

public slots:
	void TakeScreenshot();
	void FillNumParticleFinishedEditing();

	void ImportMesh();
	void AddCollider();

	void SetVelocityText(bool b, float f, float x, float y, float z);
	void SetSelectionText(QString s, bool b, int i);

	void StartSimulation();
	void StopSimulation();

	void resizeEvent(QResizeEvent* event) override;
	void moveEvent(QMoveEvent* event) override;
	void keyPressEvent(QKeyEvent* event) override;

public:
	explicit MainWindow(QWidget* parent = nullptr);
	~MainWindow();

private:
	void SetupUI();

	Ui::MainWindow* m_ui;
};

#endif