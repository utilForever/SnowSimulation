/*************************************************************************
> File Name: main.cpp
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: main entry of snow simulation.
> Created Time: 2017/03/30
> Copyright (c) 2017, Chan-Ho Chris Ohk
*************************************************************************/
#include <QtWidgets/QApplication>

#include "UI/MainWindow.h"

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	MainWindow w;
	w.show();
	return a.exec();
}
