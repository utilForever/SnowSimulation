/*************************************************************************
> File Name: ViewPanel.h
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: View panel of snow simulation.
> Created Time: 2017/12/28
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#ifndef SNOW_SIMULATION_VIEW_PANEL_H
#define SNOW_SIMULATION_VIEW_PANEL_H

#include <Geometry/Mesh.h>

#include <QGLWidget>
#include <QTimer>
#include <QElapsedTimer>
#include <QFile>
#include <QDir>

class InfoPanel;
class Viewport;
class Scene;
class Engine;
class SceneNode;
class Tool;
class SelectionTool;
class SceneIO;

class ViewPanel : public QGLWidget
{
	Q_OBJECT

public:
	ViewPanel(QWidget* parent);
	virtual ~ViewPanel();

	// Returns whether or not it started
	bool StartSimulation();
	void StopSimulation();

public slots:
	void ResetViewport();

	void initializeGL() override;
	void paintGL() override;

	void resizeEvent(QResizeEvent* event) override;

	void mousePressEvent(QMouseEvent* event) override;
	void mouseMoveEvent(QMouseEvent* event) override;
	void mouseReleaseEvent(QMouseEvent* event) override;
	void keyPressEvent(QKeyEvent* event) override;

	void ResetSimulation();
	void PauseSimulation(bool pause = true);
	void ResumeSimulation();
	void PauseDrawing();
	void ResumeDrawing();
	void UpdateColliders(float timestep);

	void LoadMesh(const QString& fileName);

	void AddCollider(int colliderType);

	void SetTool(int tool);

	void UpdateSceneGrid();

	void ClearSelection();
	void FillSelectedMesh();
	void SaveSelectedMesh();

	void OpenScene();
	void SaveScene();

	void ZeroVelocityOfSelected();
	void GiveVelocityToSelected();
	void CheckSelected();

signals:
	void ShowMeshes();
	void ShowParticles();

	void ChangeSelection(QString s, bool b, int i = 0);

	void ChangeVel(bool b, float f = 0, float x = 0, float y = 0, float z = 0);

protected:
	void PaintGrid();

	bool HasGridVBO() const;
	void BuildGridVBO();
	void DeleteGridVBO();

	QTimer m_ticker;
	QElapsedTimer m_timer;

	InfoPanel* m_infoPanel;
	Viewport* m_viewport;
	Tool* m_tool;

	SceneIO* m_sceneIO;
	SceneNode* m_selected{};

	Engine* m_engine;
	Scene* m_scene;

	GLuint m_gridVBO{};
	int m_majorSize{};
	int m_minorSize{};
	bool m_draw;
	float m_fps;
	float m_prevTime{};

	friend class Tool;
	friend class SelectionTool;
	friend class MoveTool;
	friend class RotateTool;
	friend class ScaleTool;
	friend class VelocityTool;
};

#endif