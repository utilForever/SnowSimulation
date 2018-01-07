/*************************************************************************
> File Name: SceneNodeIterator.h
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Scene node iterator of snow simulation.
> Created Time: 2018/01/07
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#ifndef SNOW_SIMULATION_SCENE_NODE_ITERATOR_H
#define SNOW_SIMULATION_SCENE_NODE_ITERATOR_H

#include <Scene/SceneNode.h>

#include <QList>

class SceneNodeIterator
{
public:
	SceneNodeIterator() : m_index(0)
	{
		// Do nothing
	}

	SceneNodeIterator(const QList<SceneNode*>& nodes) : m_nodes(nodes), m_index(0)
	{
		// Do nothing
	}

	SceneNodeIterator(const SceneNodeIterator& other) : m_nodes(other.m_nodes), m_index(other.m_index)
	{
		// Do nothing
	}

	SceneNodeIterator& operator++()
	{
		++m_index;
		return *this;
	}

	SceneNodeIterator operator++(int)
	{
		SceneNodeIterator result(*this);
		++(*this);
		return result;
	}

	SceneNode* operator*()
	{
		return m_nodes[m_index];
	}

	bool IsValid() const
	{
		return m_index < m_nodes.size();
	}
	
	void Reset()
	{
		m_index = 0;
	}

	int Size() const
	{
		return m_nodes.size();
	}

private:
	QList<SceneNode*> m_nodes;
	int m_index;
};

#endif