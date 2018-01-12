/*************************************************************************
> File Name: ViewPanel.cpp
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: View panel of snow simulation.
> Created Time: 2017/12/28
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#include <GL/glew.h>
#include <GL/gl.h>

#include <Common/Util.h>
#include <Geometry/BBox.h>
#include <Geometry/Mesh.h>
#include <IO/ObjParser.h>
#include <IO/SceneIO.h>
#include <Scene/Scene.h>
#include <Scene/SceneCollider.h>
#include <Scene/SceneGrid.h>
#include <Scene/SceneNode.h>
#include <Scene/SceneNodeIterator.h>
#include <Simulation/Engine.h>
#include <Simulation/ImplicitCollider.h>
#include <Simulation/ParticleSystem.h>
#include <UI/InfoPanel.h>
#include <UI/Picker.h>
#include <UI/UISettings.h>
#include <UI/UserInput.h>
#include <UI/ViewPanel.h>
#include <UI/Tools/VelocityTool.h>
#include <Viewport/Viewport.h>

#ifndef GLM_FORCE_RADIANS
#define GLM_FORCE_RADIANS
#endif

#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <QFileDialog>
#include <QMessageBox>
#include <QQueue>

constexpr int FPS = 30;

constexpr int MAJOR_GRID_N = 2;
constexpr float MAJOR_GRID_TICK = 0.5f;
constexpr float MINOR_GRID_TICK = 0.1f;

ViewPanel::ViewPanel(QWidget* parent) :
	QGLWidget(QGLFormat(QGL::SampleBuffers), parent),
	m_infoPanel(nullptr), m_tool(nullptr)
{
	m_viewport = new Viewport;
	ResetViewport();

	m_infoPanel = new InfoPanel(this);
	m_infoPanel->SetInfo("Major Grid Unit", QString::number(MAJOR_GRID_TICK) + " m");
	m_infoPanel->SetInfo("Minor Grid Unit", QString::number(100 * MINOR_GRID_TICK) + " cm");
	m_infoPanel->SetInfo("FPS", "XXXXXX");
	m_infoPanel->SetInfo("Sim Time", "XXXXXXX");
	m_draw = true;
	m_fps = FPS;

	m_sceneIO = new SceneIO;

	m_scene = new Scene;
	m_engine = new Engine;

	makeCurrent();
	glewInit();
}

ViewPanel::~ViewPanel()
{
	makeCurrent();
	DeleteGridVBO();

	if (m_engine != nullptr)
	{
		delete m_engine;
		m_engine = nullptr;
	}

	if (m_viewport != nullptr)
	{
		delete m_viewport;
		m_viewport = nullptr;
	}

	if (m_tool != nullptr)
	{
		delete m_tool;
		m_tool = nullptr;
	}

	if (m_infoPanel != nullptr)
	{
		delete m_infoPanel;
		m_infoPanel = nullptr;
	}

	if (m_scene != nullptr)
	{
		delete m_scene;
		m_scene = nullptr;
	}

	if (m_sceneIO != nullptr)
	{
		delete m_sceneIO;
		m_sceneIO = nullptr;
	}
}

bool ViewPanel::StartSimulation()
{
	makeCurrent();
	
	if (!m_engine->IsRunning())
	{
		m_engine->ClearColliders();

		for (SceneNodeIterator iter = m_scene->Begin(); iter.IsValid(); ++iter)
		{
			if ((*iter)->HasRenderable())
			{
				if ((*iter)->GetType() == SceneNode::Type::SIMULATION_GRID)
				{
					m_engine->SetGrid(UISettings::BuildGrid((*iter)->GetCTM()));
				}
				else if ((*iter)->GetType() == SceneNode::Type::SCENE_COLLIDER)
				{
					SceneCollider* sceneCollider = dynamic_cast<SceneCollider*>((*iter)->GetRenderable());
					ImplicitCollider& collider(*(sceneCollider->GetImplicitCollider()));
					glm::mat4 ctm = (*iter)->GetCTM();
					collider.ApplyTransformation(ctm);
					
					glm::vec3 v = (*iter)->GetRenderable()->GetWorldVelocity(ctm);
					collider.velocity = (*iter)->GetRenderable()->GetVelocityMagnitude() * v;
					
					m_engine->AddCollider(collider);
				}
			}
		}

		bool exportVol = UISettings::exportDensity() || UISettings::exportVelocity();
		if (exportVol)
		{
			SaveScene();
			
			if (!m_sceneIO->GetSceneFile().isEmpty())
			{
				QString fileName = QFileDialog::getSaveFileName(this, "Choose export destination.", "Datas/Simulation/");
				QFileInfo info(fileName);
				m_engine->InitExporter(QString("%1/%2").arg(info.absolutePath(), info.baseName()));
			}
			else
			{
				exportVol = false;
			}
		}

		return m_engine->Start(exportVol);
	}

	return false;
}

void ViewPanel::StopSimulation()
{
	m_engine->Stop();
}

void ViewPanel::ResetViewport()
{
	m_viewport->Orient(glm::vec3(1, 1, 1), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
	m_viewport->SetDimensions(width(), height());
}

void ViewPanel::initializeGL()
{
	// OpenGL states
	QGLWidget::initializeGL();

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glEnable(GL_LINE_SMOOTH);
	glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

	m_infoPanel->SetInfo("Particles", "0");

	// Render ticker
	assert(connect(&m_ticker, SIGNAL(timeout()), this, SLOT(update())));
	m_ticker.start(1000 / FPS);
	m_timer.start();
}

void ViewPanel::paintGL()
{
	glClearColor(0.20f, 0.225f, 0.25f, 1.f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glPushAttrib(GL_TRANSFORM_BIT);
	glEnable(GL_NORMALIZE);

	m_viewport->Push();
	
	m_scene->Render();
	m_scene->RenderVelocity(true);
	m_engine->Render();
	
	PaintGrid();
	
	if (m_tool)
	{
		m_tool->Render();
	}
	
	m_viewport->DrawAxis();

	m_viewport->Pop();

	if (m_draw)
	{
		static const float filter = 0.8f;
		m_fps = (1 - filter) * (1000.f / std::max(m_timer.restart(), static_cast<int64_t>(1))) + filter * m_fps;
		m_infoPanel->SetInfo("FPS", QString::number(m_fps, 'f', 2), false);
		m_infoPanel->SetInfo("Sim Time", QString::number(m_engine->GetSimulationTime(), 'f', 3) + " s", false);
	}

	float currTime = m_engine->GetSimulationTime();
	UpdateColliders(currTime - m_prevTime);
	m_prevTime = currTime;

	m_infoPanel->Render();

	glPopAttrib();
}

void ViewPanel::resizeEvent(QResizeEvent* event)
{
	QGLWidget::resizeEvent(event);
	m_viewport->SetDimensions(width(), height());
}

void ViewPanel::mousePressEvent(QMouseEvent* event)
{
	UserInput::Update(event);

	if (UserInput::IsCtrlKey())
	{
		if (UserInput::IsMouseLeft())
		{
			m_viewport->SetState(Viewport::State::TUMBLING);
		}
		else if (UserInput::IsMouseRight())
		{
			m_viewport->SetState(Viewport::State::ZOOMING);
		}
		else if (UserInput::IsMouseMiddle())
		{
			m_viewport->SetState(Viewport::State::PANNING);
		}
	}
	else
	{
		if (UserInput::IsMouseLeft())
		{
			if (m_tool)
			{
				m_tool->MousePressed();
			}
		}
	}

	update();
}

void ViewPanel::mouseMoveEvent(QMouseEvent* event)
{
	UserInput::Update(event);
   
	m_viewport->MouseMoved();

	if (m_tool)
	{
		m_tool->MouseMoved();
	}
	
	update();
}

void ViewPanel::mouseReleaseEvent(QMouseEvent* event)
{
	UserInput::Update(event);

	m_viewport->SetState(Viewport::State::IDLE);
	
	if (m_tool)
	{
		m_tool->MouseReleased();
	}
	
	update();
}

void ViewPanel::keyPressEvent(QKeyEvent* event)
{
	if (event->key() == Qt::Key_Backspace)
	{
		m_scene->DeleteSelectedNodes();
		event->accept();
	}
	else
	{
		event->setAccepted(false);
	}

	if (m_tool)
	{
		m_tool->Update();
	}
	
	update();
}

void ViewPanel::ResetSimulation()
{
	m_engine->Reset();
}

void ViewPanel::PauseSimulation(bool pause)
{
	if (pause)
	{
		m_engine->Pause();
	}
	else
	{
		m_engine->Resume();
	}
}

void ViewPanel::ResumeSimulation()
{
	m_engine->Resume();
}

void ViewPanel::PauseDrawing()
{
	m_ticker.stop();
	m_draw = false;
}

void ViewPanel::ResumeDrawing()
{
	m_ticker.start(1000 / FPS);
	m_draw = true;
}

void ViewPanel::UpdateColliders(float timestep)
{
	for (SceneNodeIterator iter = m_scene->Begin(); iter.IsValid(); ++iter)
	{
		if ((*iter)->HasRenderable())
		{
			if ((*iter)->GetType() == SceneNode::Type::SCENE_COLLIDER)
			{
				SceneCollider* c = dynamic_cast<SceneCollider*>((*iter)->GetRenderable());
				glm::vec3 v = c->GetWorldVelocity((*iter)->GetCTM());

				if (!IsEqual(c->GetVelocityMagnitude(), 0.0f))
				{
					v = glm::normalize(v);
				}

				glm::mat4 transform = glm::translate(glm::mat4(), v * c->GetVelocityMagnitude() * timestep);
				(*iter)->ApplyTransformation(transform);
			}
		}
	}
}

void ViewPanel::LoadMesh(const QString& fileName)
{

	// single obj file is associated with multiple renderable and a single scene node.
	QList<Mesh*> meshes;

	ObjParser::Load(fileName, meshes);

	ClearSelection();

	for (int i = 0; i < meshes.size(); ++i)
	{
		Mesh* mesh = meshes[i];
		mesh->SetSelected(true);
		mesh->SetType(Mesh::Type::SNOW_CONTAINER);
		
		SceneNode* node = new SceneNode(SceneNode::Type::SNOW_CONTAINER);
		node->SetRenderable(mesh);
		m_scene->GetRoot()->AddChild(node);
	}

	m_tool->Update();

	CheckSelected();
	
	if (!UISettings::showContainers())
	{
		emit ShowMeshes();
	}
}

void ViewPanel::AddCollider(int colliderType)
{
	Vector3 parameter;
	SceneNode* node = new SceneNode(SceneNode::Type::SCENE_COLLIDER);
	float r;
	
	switch (static_cast<ColliderType>(colliderType))
	{
	case ColliderType::SPHERE:
		r = SceneCollider::GetSphereRadius();
		parameter = Vector3(r, 0, 0);
		node->ApplyTransformation(glm::scale(glm::mat4(1.f), glm::vec3(r, r, r)));
		break;
	case ColliderType::HALF_PLANE:
		parameter = Vector3(0, 1, 0);
		break;
	default:
		break;
	}

	ImplicitCollider* collider = new ImplicitCollider(static_cast<ColliderType>(colliderType), Vector3(0, 0, 0), parameter, Vector3(0, 0, 0), 0.2f);
	SceneCollider* sceneCollider = new SceneCollider(collider);

	node->SetRenderable(sceneCollider);
	glm::mat4 ctm = node->GetCTM();
	sceneCollider->SetCTM(ctm);
	m_scene->GetRoot()->AddChild(node);

	ClearSelection();
	sceneCollider->SetSelected(true);

	m_tool->Update();
	CheckSelected();
}

//void ViewPanel::setTool(int tool)
//{
//    SAFE_DELETE(m_tool);
//    Tool::Type t = (Tool::Type)tool;
//    switch (t) {
//    case Tool::SELECTION:
//        m_tool = new SelectionTool(this, t);
//        break;
//    case Tool::MOVE:
//        m_tool = new MoveTool(this, t);
//        break;
//    case Tool::ROTATE:
//        m_tool = new RotateTool(this, t);
//        break;
//    case Tool::SCALE:
//        m_tool = new ScaleTool(this, t);
//        break;
//    case Tool::VELOCITY:
//        m_tool = new VelocityTool(this, t);
//        break;
//    }
//    if (m_tool) m_tool->update();
//    update();
//}
//
//void ViewPanel::updateSceneGrid()
//{
//    m_scene->updateSceneGrid();
//    if (m_tool) m_tool->update();
//    update();
//}
//
//void
//ViewPanel::clearSelection()
//{
//    for (SceneNodeIterator it = m_scene->begin(); it.isValid(); ++it) {
//        if ((*it)->hasRenderable()) {
//            (*it)->getRenderable()->setSelected(false);
//        }
//    }
//    checkSelected();
//}
//
//void ViewPanel::fillSelectedMesh()
//{
//    Mesh *mesh = new Mesh;
//    glm::vec3 currentVel;
//    float currentMag;
//    for (SceneNodeIterator it = m_scene->begin(); it.isValid(); ++it) {
//        if ((*it)->hasRenderable() &&
//            (*it)->getType() == SceneNode::SNOW_CONTAINER &&
//            (*it)->getRenderable()->isSelected()) {
//
//            Mesh *original = dynamic_cast<Mesh*>((*it)->getRenderable());
//            Mesh *copy = new Mesh(*original);
//            const glm::mat4 transformation = (*it)->getCTM();
//            copy->applyTransformation(transformation);
//            mesh->append(*copy);
//            delete copy;
//
//            currentVel = (*it)->getRenderable()->getWorldVelVec(transformation);
//            currentMag = (*it)->getRenderable()->getVelMag();
//            if (EQ(0, currentMag)) {
//                currentVel = vec3(0, 0, 0);
//            }
//            else {
//                currentVel = vec3(currentVel.x, currentVel.y, currentVel.z);
//            }
//        }
//    }
//
//    // If there's a selection, do mesh->fill...
//    if (!mesh->isEmpty()) {
//
//        makeCurrent();
//
//        ParticleSystem *particles = new ParticleSystem;
//        particles->setVelMag(currentMag);
//        particles->setVelVec(currentVel);
//        //        mesh->fill( *particles, UiSettings::fillNumParticles(), UiSettings::fillResolution(), UiSettings::fillDensity() );
//        mesh->fill(*particles, UiSettings::fillNumParticles(), UiSettings::fillResolution(), UiSettings::fillDensity(), UiSettings::materialPreset());
//        particles->setVelocity();
//        m_engine->addParticleSystem(*particles);
//        delete particles;
//
//        m_infoPanel->setInfo("Particles", QString::number(m_engine->particleSystem()->size()));
//    }
//
//    delete mesh;
//
//    if (!UiSettings::showParticles()) emit showParticles();
//}
//
//void
//ViewPanel::saveSelectedMesh()
//{
//    QList<Mesh*> meshes;
//
//    for (SceneNodeIterator it = m_scene->begin(); it.isValid(); ++it) {
//        if ((*it)->hasRenderable() &&
//            (*it)->getType() == SceneNode::SNOW_CONTAINER &&
//            (*it)->getRenderable()->isSelected()) {
//            Mesh *copy = new Mesh(*dynamic_cast<Mesh*>((*it)->getRenderable()));
//            copy->applyTransformation((*it)->getCTM());
//            meshes += copy;
//        }
//    }
//
//    // If there's a mesh selection, save it
//    if (!meshes.empty()) {
//        QString filename = QFileDialog::getSaveFileName(this, "Choose mesh file destination.", PROJECT_PATH "/data/models/");
//        if (!filename.isNull()) {
//            if (OBJParser::save(filename, meshes)) {
//                for (int i = 0; i < meshes.size(); ++i)
//                    delete meshes[i];
//                meshes.clear();
//                LOG("Mesh saved to %s", STR(filename));
//            }
//        }
//    }
//
//
//}
//
//void
//ViewPanel::openScene()
//{
//    pauseDrawing();
//    // call file dialog
//    QString filename = QFileDialog::getOpenFileName(this, "Choose Scene File Path", PROJECT_PATH "/data/scenes/");
//    if (!filename.isNull()) m_sceneIO->read(filename, m_scene, m_engine);
//    else LOG("could not open file \n");
//    resumeDrawing();
//}
//
//void
//ViewPanel::saveScene()
//{
//    pauseDrawing();
//    // this is enforced if engine->start is called and export is not checked
//    if (m_sceneIO->sceneFile().isNull()) {
//        // filename not initialized yet
//        QString filename = QFileDialog::getSaveFileName(this, "Choose Scene File Path", PROJECT_PATH "/data/scenes/");
//        if (!filename.isNull()) {
//            m_sceneIO->setSceneFile(filename);
//            m_sceneIO->write(m_scene, m_engine);
//        }
//        else {
//            QMessageBox::warning(this, "Error", "Invalid file path");
//        }
//    }
//    else {
//        m_sceneIO->write(m_scene, m_engine);
//    }
//    resumeDrawing();
//}
//
//void ViewPanel::zeroVelOfSelected() {
//    if (!m_selected) return;
//    for (SceneNodeIterator it = m_scene->begin(); it.isValid(); ++it) {
//        if ((*it)->hasRenderable() &&
//            (*it)->getType() != SceneNode::SIMULATION_GRID &&
//            (*it)->getRenderable()->isSelected()) {
//            (*it)->getRenderable()->setVelMag(0.0f);
//            (*it)->getRenderable()->setVelVec(glm::vec3(0, 0, 0));
//            (*it)->getRenderable()->updateMeshVel();
//        }
//    }
//    checkSelected();
//}
//
//void ViewPanel::giveVelToSelected() {
//    if (!m_selected) return;
//    for (SceneNodeIterator it = m_scene->begin(); it.isValid(); ++it) {
//        if ((*it)->hasRenderable() &&
//            (*it)->getType() != SceneNode::SIMULATION_GRID &&
//            (*it)->getRenderable()->isSelected()) {
//            (*it)->getRenderable()->setVelMag(1.0f);
//            (*it)->getRenderable()->setVelVec(glm::vec3(0, 1, 0));
//            (*it)->getRenderable()->updateMeshVel();
//        }
//    }
//    checkSelected();
//}
//
//void ViewPanel::checkSelected() {
//    int counter = 0;
//    for (SceneNodeIterator it = m_scene->begin(); it.isValid(); ++it) {
//        if ((*it)->hasRenderable() && (*it)->getRenderable()->isSelected()) {
//            counter++;
//            m_selected = (*it);
//        }
//    }
//    if (counter == 0) {
//        emit changeVel(false);
//        emit changeSelection("Currently Selected: none", false);
//        m_selected = NULL;
//    }
//    else if (counter == 1 && m_selected->getType() != SceneNode::SIMULATION_GRID) {
//        glm::vec3 v;// = glm::vec3(m_selected->getRenderable()->getVelVec());
//                    //       glm::vec4 vWorld = m_selected->getCTM()*glm::vec4((v),1);
//                    //       float mag = glm::length(v);
//        if EQ(m_selected->getRenderable()->getVelMag(), 0)
//        {
//            emit changeVel(true, m_selected->getRenderable()->getVelMag(), 0, 0, 0);
//        }
//        else
//        {
//            //           v = glm::normalize(glm::vec3(vWorld.x,vWorld.y,vWorld.z));
//            v = m_selected->getRenderable()->getWorldVelVec(m_selected->getCTM());
//            emit changeVel(true, m_selected->getRenderable()->getVelMag(), v.x, v.y, v.z);
//        }
//        emit changeSelection("Currently Selected: ", true, m_selected->getType());
//    }
//    else if (counter == 1 && m_selected->getType() == SceneNode::SIMULATION_GRID) {
//        emit changeVel(false);
//        emit changeSelection("Currently Selected: Grid", false);
//    }
//    else {
//        emit changeVel(false);
//        emit changeSelection("Currently Selected: more than one object", false);
//        m_selected = NULL;
//    }
//}
//
//// Paint grid on XZ plane to orient viewport
//void
//ViewPanel::paintGrid()
//{
//    if (!hasGridVBO()) buildGridVBO();
//
//    glPushAttrib(GL_COLOR_BUFFER_BIT);
//    glEnable(GL_BLEND);
//    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
//    glEnable(GL_LINE_SMOOTH);
//    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
//    glBindBuffer(GL_ARRAY_BUFFER, m_gridVBO);
//    glEnableClientState(GL_VERTEX_ARRAY);
//    glVertexPointer(3, GL_FLOAT, sizeof(vec3), (void*)(0));
//    glColor4f(0.5f, 0.5f, 0.5f, 0.8f);
//    glLineWidth(2.5f);
//    glDrawArrays(GL_LINES, 0, 4);
//    glColor4f(0.5f, 0.5f, 0.5f, 0.65f);
//    glLineWidth(1.5f);
//    glDrawArrays(GL_LINES, 4, m_majorSize - 4);
//    glColor4f(0.5f, 0.5f, 0.5f, 0.5f);
//    glLineWidth(0.5f);
//    glDrawArrays(GL_LINES, m_majorSize, m_minorSize);
//    glBindBuffer(GL_ARRAY_BUFFER, 0);
//    glDisableClientState(GL_VERTEX_ARRAY);
//    glEnd();
//    glPopAttrib();
//}
//
//bool
//ViewPanel::hasGridVBO() const
//{
//    return m_gridVBO > 0 && glIsBuffer(m_gridVBO);
//}
//
//void
//ViewPanel::buildGridVBO()
//{
//    deleteGridVBO();
//
//    QVector<vec3> data;
//
//    static const int minorN = MAJOR_GRID_N * MAJOR_GRID_TICK / MINOR_GRID_TICK;
//    static const float max = MAJOR_GRID_N * MAJOR_GRID_TICK;
//    for (int i = 0; i <= MAJOR_GRID_N; ++i) {
//        float x = MAJOR_GRID_TICK * i;
//        data += vec3(x, 0.f, -max);
//        data += vec3(x, 0.f, max);
//        data += vec3(-max, 0.f, x);
//        data += vec3(max, 0.f, x);
//        if (i) {
//            data += vec3(-x, 0.f, -max);
//            data += vec3(-x, 0.f, max);
//            data += vec3(-max, 0.f, -x);
//            data += vec3(max, 0.f, -x);
//        }
//    }
//    m_majorSize = data.size();
//
//    for (int i = -minorN; i <= minorN; ++i) {
//        float x = MINOR_GRID_TICK * i;
//        data += vec3(x, 0.f, -max);
//        data += vec3(x, 0.f, max);
//        data += vec3(-max, 0.f, x);
//        data += vec3(max, 0.f, x);
//    }
//    m_minorSize = data.size() - m_majorSize;
//
//    glGenBuffers(1, &m_gridVBO);
//    glBindBuffer(GL_ARRAY_BUFFER, m_gridVBO);
//    glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(vec3), data.data(), GL_STATIC_DRAW);
//    glBindBuffer(GL_ARRAY_BUFFER, 0);
//}
//
//void
//ViewPanel::deleteGridVBO()
//{
//    if (hasGridVBO()) {
//        glDeleteBuffers(1, &m_gridVBO);
//    }
//    m_gridVBO = 0;
//}