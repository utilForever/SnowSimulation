/*************************************************************************
> File Name: Picker.cpp
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Picker of snow simulation.
> Created Time: 2018/01/10
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#include <UI/Picker.h>

#include <GL/glew.h>
#include <GL/gl.h>

const unsigned int Picker::NO_PICK = INT_MAX;

Picker::Picker(int n) : m_nObjects(n), m_picks(nullptr)
{
	if (m_nObjects > 0)
	{
		m_picks = new PickRecord[m_nObjects];

		glSelectBuffer(4 * m_nObjects, reinterpret_cast<unsigned int*>(m_picks));
		glRenderMode(GL_SELECT);
		glInitNames();
		glPushName(0);
		
		m_selectMode = true;
	}
}

Picker::~Picker()
{
	if (m_picks != nullptr)
	{
		delete m_picks;
		m_picks = nullptr;
	}
}

void Picker::SetObjectIndex(unsigned int i) const
{
	glLoadName(i);
}

unsigned int Picker::GetPick()
{
	unsigned int index = NO_PICK;
	
	if (m_nObjects > 0 && m_selectMode == true)
	{
		int hits = glRenderMode(GL_RENDER);
		unsigned int depth = ~0;

		for (int i = 0; i < hits; ++i)
		{
			const PickRecord& pick = m_picks[i];
			
			if (pick.minDepth < depth)
			{
				index = pick.name;
				depth = pick.minDepth;
			}
		}

		m_selectMode = false;
	}

	return index;
}

QList<unsigned int> Picker::GetPicks()
{
	QList<unsigned int> picks;

	if (m_nObjects > 0 && m_selectMode == true)
	{
		int hits = glRenderMode(GL_RENDER);

		for (int i = 0; i < hits; ++i)
		{
			const PickRecord& pick = m_picks[i];
			picks += pick.name;
		}

		m_selectMode = false;
	}

	return picks;
}