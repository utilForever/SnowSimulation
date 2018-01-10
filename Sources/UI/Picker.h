/*************************************************************************
> File Name: Picker.h
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Picker of snow simulation.
> Created Time: 2018/01/10
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#ifndef SNOW_SIMULATION_PICKER_H
#define SNOW_SIMULATION_PICKER_H

#include <QList>

class Picker
{
public:
	static const unsigned int NO_PICK;

	Picker(int n);
	~Picker();

	void SetObjectIndex(unsigned int i) const;

	unsigned int GetPick();
	QList<unsigned int> GetPicks();

private:
	struct PickRecord
	{
		unsigned int numNames;
		unsigned int minDepth;
		unsigned int maxDepth;
		unsigned int name;
	};

	bool m_selectMode;

	int m_nObjects;
	PickRecord* m_picks;
};

#endif