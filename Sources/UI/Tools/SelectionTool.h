/*************************************************************************
> File Name: SelectionTool.h
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Selection tool of snow simulation.
> Created Time: 2018/01/10
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#ifndef SNOW_SIMULATION_SELECTION_TOOL_H
#define SNOW_SIMULATION_SELECTION_TOOL_H

#include <UI/Tools/Tool.h>

class SceneNode;

struct Vector3;

class SelectionTool : public Tool
{
public:
	SelectionTool(ViewPanel* panel, Type t);
	virtual ~SelectionTool();

	void MousePressed() override;
	void MouseReleased() override;

	void Update() override;

	void Render() override;

	bool HasSelection(Vector3& center) const;
	bool HasRotatableSelection(Vector3& center) const;
	bool HasScalableSelection(Vector3& center) const;

	void ClearSelection();
	SceneNode* GetSelectedSceneNode();
};

#endif