/*************************************************************************
> File Name: MainWindow.cpp
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Main window UI of snow simulation.
> Created Time: 2017/06/11
> Copyright (c) 2017, Chan-Ho Chris Ohk
*************************************************************************/
#include "MainWindow.h"
#include "ui_MainWindow.h"

MainWindow::MainWindow(QWidget *parent) :
	QMainWindow(parent), m_ui(new Ui::MainWindow)
{
	m_ui->setupUi(this);

	SetupUI();

	this->setWindowTitle("Snow Simulation");
}

MainWindow::~MainWindow()
{
	delete m_ui;
	m_ui = nullptr;
}

void MainWindow::SetupUI()
{
	
}