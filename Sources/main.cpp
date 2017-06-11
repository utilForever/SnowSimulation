/*************************************************************************
> File Name: main.cpp
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: main entry of snow simulation.
> Created Time: 2017/03/30
> Copyright (c) 2017, Chan-Ho Chris Ohk
*************************************************************************/
#include <iostream>

#include "UI/MainWindow.h"

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	MainWindow w;

	if (argc < 2)
	{
		w.show();
		return a.exec();
	}

	std::cout << "Unknown argument " << argv[1] <<
		", Run with empty argument list to run with GUI.\n";
	
	return 0;
}
