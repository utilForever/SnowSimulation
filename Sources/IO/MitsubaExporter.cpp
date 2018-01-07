/*************************************************************************
> File Name: MitsubaExporter.cpp
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Mitsuba exporter of snow simulation.
> Created Time: 2018/01/07
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#include <Common/Math.h>
#include <IO/MitsubaExporter.h>
#include <UI/UISettings.h>

#include "scene/scenenode.h"
#include "geometry/bbox.h"
#include "scene/scene.h"
#include "sim/particle.h"
#include "sim/engine.h"
#include "viewport/camera.h"

#include <QFile>
#include <QtConcurrent/QtConcurrent>

#include <fstream>
#include <iomanip>
#include <iostream>

#include <stdio.h>