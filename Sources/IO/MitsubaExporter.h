/*************************************************************************
> File Name: MitsubaExporter.h
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Mitsuba exporter of snow simulation.
> Created Time: 2018/01/07
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#ifndef SNOW_SIMULATION_MITSUBA_EXPORTER_H
#define SNOW_SIMULATION_MITSUBA_EXPORTER_H

#include <Geometry/BBox.h>
#include <Geometry/Grid.h>
#include <Simulation/Node.h>

#include <QString>
#include <QtXml/QtXml>

#include <glm/geometric.hpp>

class ImplicitCollider;
class SceneNode;

class MitsubaExporter
{
public:
    MitsubaExporter();
    MitsubaExporter(QString fPrefix, int fps);
    ~MitsubaExporter();

    float GetSPF();
    float GetLastUpdateTime();
    void Reset(Grid grid);
    Node* GetNodesPtr();
    void RunExportThread(float t);
    void ExportScene(float t);

private:
    void ExportDensityData(float t);
    void ExportVelocityData(float t);
    void Init();

    void WriteVOLHeader(std::ofstream& os, int channels);

    // file format prefix this is written to, i.e. m_fileprefix = /home/evjang/teapot
    QString m_filePrefix;

    float m_lastUpdateTime;
    // number of frames to export every second of simulation
    int m_fps;
    // seconds per frame
    float m_spf;
    Node* m_nodes;
    Grid m_grid;
    int m_frame;
    bool m_busy;

    // promise object used with QtConcurrentRun
    QFuture<void> m_future;
};

#endif