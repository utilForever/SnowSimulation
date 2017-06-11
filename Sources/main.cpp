#include "SnowSimulation.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	SnowSimulation w;
	w.show();
	return a.exec();
}
