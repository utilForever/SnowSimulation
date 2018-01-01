/*************************************************************************
> File Name: Node.h
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Node structure of snow simulation.
> Created Time: 2018/01/01
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#ifndef SNOW_SIMULATION_NODE_H
#define SNOW_SIMULATION_NODE_H

#include <CUDA/Matrix.h>
#include <CUDA/Vector.h>
#include <Geometry/Grid.h>

struct Node
{
	float mass;
	Vector3 velocity;
	// v_(n+1) - v_n (store this value through steps 4, 5, 6)
	Vector3 velocityChange;
	Vector3 force;

	Node() : mass(0), velocity(0, 0, 0), force(0, 0, 0)
	{
		// Do nothing
	}
};

#endif