/*************************************************************************
> File Name: MainWindow.cpp
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Main window UI of snow simulation.
> Created Time: 2017/06/11
> Copyright (c) 2017, Chan-Ho Chris Ohk
*************************************************************************/
#include <UI/DataBinding.h>
#include <UI/MainWindow.h>
#include <UI/UISettings.h>
#include <UI/UserInput.h>

#include "ui_MainWindow.h"

MainWindow::MainWindow(QWidget *parent) :
	QMainWindow(parent), m_ui(new Ui::MainWindow)
{
	UISettings::LoadSettings();

	m_ui->setupUi(this);

	SetupUI();

	this->setWindowTitle("Snow Simulation");
	this->move(UISettings::windowPosition());
	this->resize(UISettings::windowSize());
}

MainWindow::~MainWindow()
{
	UserInput::DeleteInstance();

	delete m_ui;
	m_ui = nullptr;

	UISettings::SaveSettings();
}

void MainWindow::SetupUI()
{
		
}

void MainWindow::StartSimulation()
{
	
}

void MainWindow::StopSimulation()
{
	
}