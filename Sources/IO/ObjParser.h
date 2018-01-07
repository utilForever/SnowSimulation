/*************************************************************************
> File Name: ObjParser.h
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Obj file parser of snow simulation.
> Created Time: 2018/01/07
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#ifndef SNOW_SIMULATION_OBJ_PARSER_H
#define SNOW_SIMULATION_OBJ_PARSER_H

#include <Geometry/Mesh.h>

#include <QFile>
#include <QString>
#include <QQueue>
#include <QList>
#include <QVector>

class ObjParser
{
public:
    static void Load(const QString& fileName, QList<Mesh*>& meshes);
    static bool Save(const QString& fileName, QList<Mesh*>& meshes);

    ObjParser(const QString& fileName = QString());
    virtual ~ObjParser();

    QString GetFileName() const;
    void SetFileName(const QString& fileName);

    Mesh* PopMesh();
    bool HasMeshes() const;
    void SetMeshes(const QList<Mesh*>& meshes);

    bool Load();
    bool Save();
    void Clear();

private:
    enum class Mode
    {
        VERTEX,
        FACE,
        GROUP
    };

    bool IsMeshPending() const;
    void AddMesh();

    void SetMode(Mode mode);

    bool Parse(const QStringList& lines, int& lineIndex);
    bool ParseName(const QString& line);
    bool ParseVertex(const QString& line);
    bool ParseFace(const QString& line);

    QString Write(Mesh* mesh) const;

    Mode m_mode;

    QFile m_file;

    QString m_currentName;
    QVector<Vertex> m_vertexPool;
    QVector<Mesh::Tri> m_triPool;
    QVector<Normal> m_normalPool;

    QQueue<Mesh*> m_meshes;
};

#endif