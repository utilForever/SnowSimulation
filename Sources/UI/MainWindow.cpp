/*************************************************************************
> File Name: MainWindow.cpp
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Main window UI of snow simulation.
> Created Time: 2017/06/11
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#include <Scene/Scene.h>
#include <UI/CollapsibleBox.h>
#include <UI/DataBinding.h>
#include <UI/MainWindow.h>
#include <UI/UISettings.h>
#include <UI/UserInput.h>
#include <UI/ViewPanel.h>
#include <UI/Tools/Tool.h>

#include <QFileDialog>
#include <QDir>
#include <QPixmap>
#include <iostream>

#include "ui_MainWindow.h"

MainWindow::MainWindow(QWidget* parent) :
	QMainWindow(parent), m_ui(new Ui::MainWindow)
{
	UISettings::LoadSettings();

	m_ui->setupUi(this);

	SetupUI();

	this->setWindowTitle("Snow Simulation");
	this->move(UISettings::windowPosition());
	this->resize(UISettings::windowSize());
}

MainWindow::~MainWindow()
{
	UserInput::DeleteInstance();

	delete m_ui;
	m_ui = nullptr;

	UISettings::SaveSettings();
}

void MainWindow::TakeScreenshot()
{
	// this has issues rasterizing particles...
	m_ui->viewPanel->PauseDrawing();
	m_ui->viewPanel->PauseSimulation();

	QPixmap pixmap(this->rect().size());
	this->render(&pixmap, QPoint(), QRegion(this->rect()));
	
	// prompt user where to save it
	QString fileName = QFileDialog::getSaveFileName(this, QString("Save Screenshot"), "Datas/");
	if (!fileName.isEmpty())
	{
		QFile file(fileName);
		file.open(QIODevice::WriteOnly);
		pixmap.save(&file, "PNG");
		file.close();
	}

	m_ui->viewPanel->ResumeDrawing();
	m_ui->viewPanel->ResumeSimulation();
}

void MainWindow::FillNumParticleFinishedEditing()
{
	// rounds number of particles to nearest multiple of 512
	const int n = m_ui->fillNumParticlesSpinbox->value();
	int numParticles = (n / 512) * 512;
	numParticles += 512 * (n < 512);

	if (numParticles != n)
	{
		m_ui->fillNumParticlesSpinbox->setValue(numParticles);
	}
}

void MainWindow::ImportMesh()
{
	m_ui->viewPanel->PauseSimulation();
	m_ui->viewPanel->PauseDrawing();

	QString fileName = QFileDialog::getOpenFileName(this, "Select mesh to import.", "Datas/Models", "*.obj");
	if (!fileName.isEmpty())
	{
		m_ui->viewPanel->LoadMesh(fileName);
	}

	m_ui->viewPanel->ResumeSimulation();
	m_ui->viewPanel->ResumeDrawing();
}

void MainWindow::AddCollider()
{
	m_ui->viewPanel->AddCollider(m_ui->chooseCollider->currentIndex());
}

void MainWindow::SetVelocityText(bool b, float f, float x, float y, float z)
{
	if (!b)
	{
		m_ui->velLabel->setText("Velocity: ");
		return;
	}

	glm::vec3 v = Vector3(x, y, z);
	v = v * f;
	
	QString toSet = "Velocity: (";

	QString s1 = QString::number(v.x, 'g', 2);
	toSet.append(s1);
	toSet.append(" m/s,");
	QString s2 = QString::number(v.y, 'g', 2);
	toSet.append(s2);
	toSet.append(" m/s,");
	QString s3 = QString::number(v.z, 'g', 2);
	toSet.append(s3);
	toSet.append(" m/s)");

	m_ui->velLabel->setText(toSet);
}

void MainWindow::SetSelectionText(QString s, bool b, int i)
{
	if (!b)
	{
		m_ui->currentlySelectedLabel->setText(s);
		return;
	}

	switch (i)
	{
	case 1:
		m_ui->currentlySelectedLabel->setText("Currently Selected: Collider");
		break;
	case 2:
		m_ui->currentlySelectedLabel->setText("Currently Selected: Snow Container");
		break;
	default:
		break;
	}
}

void MainWindow::StartSimulation()
{
	if (m_ui->viewPanel->StartSimulation())
	{
		m_ui->viewPanel->ClearSelection();
		m_ui->selectionToolButton->click();
		m_ui->startButton->setEnabled(false);
		m_ui->stopButton->setEnabled(true);
		m_ui->pauseButton->setEnabled(true);
		m_ui->resetButton->setEnabled(false);
	}
}

void MainWindow::StopSimulation()
{
	m_ui->viewPanel->StopSimulation();
	m_ui->startButton->setEnabled(true);
	m_ui->stopButton->setEnabled(false);
	if (m_ui->pauseButton->isChecked())
	{
		m_ui->pauseButton->click();
	}
	m_ui->pauseButton->setEnabled(false);
	m_ui->resetButton->setEnabled(true);
}

void MainWindow::resizeEvent(QResizeEvent*)
{
	UISettings::windowSize() = size();
}

void MainWindow::moveEvent(QMoveEvent*)
{
	UISettings::windowPosition() = pos();
}

void MainWindow::keyPressEvent(QKeyEvent* event)
{
	if (event->key() == Qt::Key_Q)
	{
		m_ui->selectionToolButton->click();
		event->accept();
	}
	else if (event->key() == Qt::Key_W)
	{
		m_ui->moveToolButton->click();
		event->accept();
	}
	else if (event->key() == Qt::Key_E)
	{
		m_ui->rotateToolButton->click();
		event->accept();
	}
	else if (event->key() == Qt::Key_R)
	{
		m_ui->scaleToolButton->click();
		event->accept();
	}
	else
	{
		event->setAccepted(false);
	}
}

void MainWindow::SetupUI()
{
	assert(connect(m_ui->actionSave_Mesh, SIGNAL(triggered()), m_ui->viewPanel, SLOT(SaveSelectedMesh())));
	assert(connect(m_ui->actionOpen_Scene, SIGNAL(triggered()), m_ui->viewPanel, SLOT(OpenScene())));
	assert(connect(m_ui->actionSave_Scene, SIGNAL(triggered()), m_ui->viewPanel, SLOT(SaveScene())));

	// Mesh Filling
	assert(connect(m_ui->importButton, SIGNAL(clicked()), this, SLOT(ImportMesh())));
	assert(connect(m_ui->fillButton, SIGNAL(clicked()), m_ui->viewPanel, SLOT(FillSelectedMesh())));
	FloatBinding::BindSpinBox(m_ui->fillResolutionSpinbox, UISettings::fillResolution(), this);
	IntBinding::BindSpinBox(m_ui->fillNumParticlesSpinbox, UISettings::fillNumParticles(), this);
	assert(connect(m_ui->fillNumParticlesSpinbox, SIGNAL(editingFinished()), this, SLOT(FillNumParticleFinishedEditing())));
	FloatBinding::BindSpinBox(m_ui->densitySpinbox, UISettings::fillDensity(), this);
	ComboIntAttribute::BindInt(m_ui->snowMaterialCombo, &UISettings::materialPreset(), this);
	assert(connect(m_ui->meshGiveVelocityButton, SIGNAL(clicked()), m_ui->viewPanel, SLOT(GiveVelocityToSelected())));
	assert(connect(m_ui->MeshZeroVelocityButton, SIGNAL(clicked()), m_ui->viewPanel, SLOT(ZeroVelocityOfSelected())));

	// Simulation
	assert(connect(m_ui->startButton, SIGNAL(clicked()), this, SLOT(StartSimulation())));
	assert(connect(m_ui->stopButton, SIGNAL(clicked()), this, SLOT(StopSimulation())));
	assert(connect(m_ui->pauseButton, SIGNAL(toggled(bool)), m_ui->viewPanel, SLOT(PauseSimulation(bool))));
	assert(connect(m_ui->resetButton, SIGNAL(clicked()), m_ui->viewPanel, SLOT(ResetSimulation())));
	IntBinding::BindSpinBox(m_ui->gridXSpinbox, UISettings::gridDimensions().x, this);
	IntBinding::BindSpinBox(m_ui->gridYSpinbox, UISettings::gridDimensions().y, this);
	IntBinding::BindSpinBox(m_ui->gridZSpinbox, UISettings::gridDimensions().z, this);
	FloatBinding::BindSpinBox(m_ui->gridResolutionSpinbox, UISettings::gridResolution(), this);
	assert(connect(m_ui->gridXSpinbox, SIGNAL(valueChanged(int)), m_ui->viewPanel, SLOT(UpdateSceneGrid())));
	assert(connect(m_ui->gridYSpinbox, SIGNAL(valueChanged(int)), m_ui->viewPanel, SLOT(UpdateSceneGrid())));
	assert(connect(m_ui->gridZSpinbox, SIGNAL(valueChanged(int)), m_ui->viewPanel, SLOT(UpdateSceneGrid())));
	assert(connect(m_ui->gridResolutionSpinbox, SIGNAL(valueChanged(double)), m_ui->viewPanel, SLOT(UpdateSceneGrid())));
	FloatBinding::BindSpinBox(m_ui->timeStepSpinbox, UISettings::timeStep(), this);
	BoolBinding::BindCheckBox(m_ui->implicitCheckbox, UISettings::implicit(), this);

	// exporting
	BoolBinding::BindCheckBox(m_ui->exportDensityCheckbox, UISettings::exportDensity(), this);
	BoolBinding::BindCheckBox(m_ui->exportVelocityCheckbox, UISettings::exportVelocity(), this);
	IntBinding::BindSpinBox(m_ui->exportFPSSpinBox, UISettings::exportFPS(), this);
	FloatBinding::BindSpinBox(m_ui->maxTimeSpinBox, UISettings::maxTime(), this);

	// SceneCollider
	assert(connect(m_ui->colliderAddButton, SIGNAL(clicked()), this, SLOT(AddCollider())));

	// View Panel
	assert(connect(m_ui->showContainersCheckbox, SIGNAL(toggled(bool)), m_ui->showContainersCombo, SLOT(setEnabled(bool))));
	assert(connect(m_ui->showCollidersCheckbox, SIGNAL(toggled(bool)), m_ui->showCollidersCombo, SLOT(setEnabled(bool))));
	assert(connect(m_ui->showGridCheckbox, SIGNAL(toggled(bool)), m_ui->showGridCombo, SLOT(setEnabled(bool))));
	assert(connect(m_ui->showGridDataCheckbox, SIGNAL(toggled(bool)), m_ui->showGridDataCombo, SLOT(setEnabled(bool))));
	assert(connect(m_ui->showParticlesCheckbox, SIGNAL(toggled(bool)), m_ui->showParticlesCombo, SLOT(setEnabled(bool))));
	assert(connect(m_ui->viewPanel, SIGNAL(ShowParticles()), m_ui->showParticlesCheckbox, SLOT(click())));
	assert(connect(m_ui->viewPanel, SIGNAL(ShowMeshes()), m_ui->showContainersCheckbox, SLOT(click())));
	CheckboxBoolAttribute::BindBool(m_ui->showContainersCheckbox, &UISettings::showContainers(), this);
	ComboIntAttribute::BindInt(m_ui->showContainersCombo, &UISettings::showContainersMode(), this);
	CheckboxBoolAttribute::BindBool(m_ui->showCollidersCheckbox, &UISettings::showColliders(), this);
	ComboIntAttribute::BindInt(m_ui->showCollidersCombo, &UISettings::showCollidersMode(), this);
	CheckboxBoolAttribute::BindBool(m_ui->showGridCheckbox, &UISettings::showGrid(), this);
	ComboIntAttribute::BindInt(m_ui->showGridCombo, &UISettings::showGridMode(), this);
	CheckboxBoolAttribute::BindBool(m_ui->showGridDataCheckbox, &UISettings::showGridData(), this);
	ComboIntAttribute::BindInt(m_ui->showGridDataCombo, &UISettings::showGridDataMode(), this);
	CheckboxBoolAttribute::BindBool(m_ui->showParticlesCheckbox, &UISettings::showParticles(), this);
	ComboIntAttribute::BindInt(m_ui->showParticlesCombo, &UISettings::showParticlesMode(), this);

	// Tools
	m_ui->toolButtonGroup->setId(m_ui->selectionToolButton, static_cast<int>(Tool::Type::SELECTION));
	m_ui->toolButtonGroup->setId(m_ui->moveToolButton, static_cast<int>(Tool::Type::MOVE));
	m_ui->toolButtonGroup->setId(m_ui->rotateToolButton, static_cast<int>(Tool::Type::ROTATE));
	m_ui->toolButtonGroup->setId(m_ui->scaleToolButton, static_cast<int>(Tool::Type::SCALE));
	m_ui->toolButtonGroup->addButton(m_ui->velocityToolButton);
	m_ui->toolButtonGroup->setId(m_ui->velocityToolButton, static_cast<int>(Tool::Type::VELOCITY));
	assert(connect(m_ui->toolButtonGroup, SIGNAL(buttonClicked(int)), m_ui->viewPanel, SLOT(SetTool(int))));
	m_ui->selectionToolButton->click();

	// Selected Object
	assert(connect(m_ui->viewPanel, SIGNAL(ChangeVelocity(bool, float, float, float, float)), this, SLOT(SetVelocityText(bool, float, float, float, float))));
	assert(connect(m_ui->viewPanel, SIGNAL(ChangeSelection(QString, bool, int)), this, SLOT(SetSelectionText(QString, bool, int))));

	m_ui->toolGroupBox->Init();
	m_ui->SelectedObjectGroupBox->Init();
	m_ui->snowContainersGroupBox->Init();
	m_ui->simulationGroupBox->Init();
	m_ui->gridGroupBox->Init();
	m_ui->exportGroupBox->Init();
	m_ui->parametersGroupBox->Init();
	m_ui->collidersGroupBox->Init();
	m_ui->viewPanelGroupBox->Init();
}