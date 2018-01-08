/*************************************************************************
> File Name: SceneIO.cpp
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Scene IO of snow simulation.
> Created Time: 2018/01/08
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#include <Common/Util.h>
#include <CUDA/Vector.h>
#include <Geometry/Mesh.h>
#include <IO/SceneIO.h>
#include <Scene/Scene.h>
#include <Scene/SceneCollider.h>
#include <Scene/SceneNodeIterator.h>
#include <Simulation/Engine.h>
#include <Simulation/ParticleSystem.h>
#include <UI/UISettings.h>

#include <QFileDialog>
#include <QMessageBox>

#include <glm/gtx/string_cast.hpp>

SceneIO::SceneIO()
{
    // Do nothing
}

bool SceneIO::Read(QString fileName, Scene* scene, Engine* engine)
{
    m_document.clear();

    QFileInfo info(fileName);
    m_sceneFilePrefix = QString("%1/%2").arg(info.absolutePath(), info.baseName());

    QFile* file = new QFile(fileName);
    if (!file->open(QIODevice::ReadOnly | QIODevice::Text))
    {
        QMessageBox msgBox;
        msgBox.setText("Error : Invalid XML file");
        msgBox.exec();
        return false;
    }

    engine->Reset();
    scene->Reset();
    scene->InitSceneGrid();

    QString errMsg;
    int errLine;
    int errCol;
    if (!m_document.setContent(file, &errMsg, &errLine, &errCol))
    {
        QMessageBox msgBox;
        errMsg = QString("XML Import Error : Line %1, Col %2 : %3").arg(QString::number(errLine), QString::number(errCol), errMsg);
        msgBox.setText(errMsg);
        msgBox.exec();
        return false;
    }

    ApplySimulationParameters();
    ApplyExportSettings();
    ApplyParticleSystem(scene);
    ApplyGrid(scene);
    ApplyColliders(scene, engine);
}

bool SceneIO::Write(Scene* scene, Engine* engine)
{
    m_document.clear();
    QDomProcessingInstruction processInstruct = m_document.createProcessingInstruction("xml", "version=\"1.0\" encoding=\"utf-8\" ");
    m_document.appendChild(processInstruct);

    // root node of the scene
    QDomElement root = m_document.createElement("SnowSimulation");
    m_document.appendChild(root);

    AppendSimulationParameters(root, UISettings::timeStep());
    AppendExportSettings(root);
    AppendParticleSystem(root, scene);
    AppendGrid(root, scene);
    AppendColliders(root, scene);

    QString fileName = QString("%1.xml").arg(m_sceneFilePrefix);
    QFile file(fileName);
    if (!file.open(QIODevice::ReadWrite | QIODevice::Truncate | QIODevice::Text))
    {
        LOG("write failed!");
    }
    else
    {
        QTextStream stream(&file);
        int indent = 4;

        stream << m_document.toString(indent);
        file.close();
        
        LOG("file written!");
    }
}

QString SceneIO::GetSceneFile()
{
    return m_sceneFilePrefix;
}

void SceneIO::SetSceneFile(QString fileName)
{
    // turn fileName into an absolute path
    QFileInfo info(fileName);
    m_sceneFilePrefix = QString("%1/%2").arg(info.absolutePath(), info.baseName());
}

void SceneIO::ApplySimulationParameters()
{
    QDomNodeList nlist = m_document.elementsByTagName("SimulationParameters");
    QDomElement sNode = nlist.at(0).toElement();
    QDomNodeList sList = sNode.childNodes();

    for (int i = 0; i < sList.size(); ++i)
    {
        QDomElement n = sList.at(i).toElement();

        if (n.attribute("name").compare("timeStep") == 0)
        {
            bool ok;
            float ts = n.attribute("value").toFloat(&ok);
            
            if (ok == true)
            {
                UISettings::timeStep() = ts;
            }
        }
    }
}

void SceneIO::ApplyExportSettings()
{
    QDomNodeList list = m_document.elementsByTagName("ExportSettings");
    QDomElement s = list.at(0).toElement();
    
    for (int i = 0; i < s.childNodes().size(); ++i)
    {
        QDomElement e = s.childNodes().at(i).toElement();
        QString name = e.attribute("name");

        if (name.compare("filePrefix") == 0)
        {
            m_sceneFilePrefix = e.attribute("value");
        }
        else if (name.compare("maxTime") == 0)
        {
            UISettings::maxTime() = e.attribute("value").toInt();
        }
        else if (name.compare("exportFPS") == 0)
        {
            UISettings::exportFPS() = e.attribute("value").toInt();
        }
        else if (name.compare("exportDensity") == 0)
        {
            UISettings::exportVelocity() = e.attribute("value").toInt();
        }
        else if (name.compare("exportVelocity") == 0)
        {
            UISettings::exportVelocity() = e.attribute("value").toInt();
        }
    }
}

void SceneIO::ApplyParticleSystem(Scene* scene)
{
    // does not call fillParticles for the user.
    QDomNodeList list = m_document.elementsByTagName("SnowContainer");
    QString fileName;
    glm::mat4 CTM;
    
    for (int s = 0; s < list.size(); ++s)
    {
        // for each SnowContainer, import the obj into the scene
        QDomElement p = list.at(s).toElement();

        for (int t = 0; t < p.childNodes().size(); ++t)
        {
            QDomElement d = p.childNodes().at(t).toElement();
            QString name = d.attribute("name");
            
            if (name.compare("fileName") == 0)
            {
                fileName = d.attribute("value");
            }
            else if (name.compare("CTM") == 0)
            {
                QStringList floatWords = d.attribute("value").split(QRegExp("\\s+"));
                int k = 0;
                
                for (int i = 0; i < 4; i++)
                {
                    for (int j = 0; j < 4; j++, k++)
                    {
                        CTM[j][i] = floatWords.at(k).toFloat();
                    }
                }
            }
        }

        scene->LoadMesh(fileName, CTM);
    }
}

void SceneIO::ApplyGrid(Scene* scene)
{
    Grid grid;
    QDomNodeList list = m_document.elementsByTagName("Grid");
    QDomElement g = list.at(0).toElement();

    for (int i = 0; i < g.childNodes().size(); ++i)
    {
        QDomElement e = g.childNodes().at(i).toElement();
        QString name = e.attribute("name");
        
        if (name.compare("gridDim") == 0)
        {
            UISettings::gridDimensions() = glm::ivec3(e.attribute("x").toInt(), e.attribute("y").toInt(), e.attribute("z").toInt());
        }
        else if (name.compare("pos") == 0)
        {
            UISettings::gridPosition() = Vector3(e.attribute("x").toFloat(), e.attribute("y").toFloat(), e.attribute("z").toFloat());
        }
        else if (name.compare("h") == 0)
        {
            UISettings::gridResolution() = e.attribute("value").toFloat();
        }
    }
    scene->UpdateSceneGrid();
}

void SceneIO::ApplyColliders(Scene* scene, Engine* engine)
{
    Vector3 center, velocity, param;
    QDomNodeList list = m_document.elementsByTagName("Collider");

    for (int i = 0; i < list.size(); ++i)
    {
        QDomElement e = list.at(i).toElement();
        int colliderType = e.attribute("type").toInt();
        
        for (int j = 0; j < e.childNodes().size(); j++)
        {
            QDomElement c = e.childNodes().at(j).toElement();
            Vector3 vector;
            vector.x = c.attribute("x").toFloat();
            vector.y = c.attribute("y").toFloat();
            vector.z = c.attribute("z").toFloat();
            QString name = c.attribute("name");

            if (name.compare("center") == 0)
            {
                center = vector;
            }
            else if (name.compare("velocity") == 0)
            {
                velocity = vector;
            }
            else if (name.compare("param") == 0)
            {
                param = vector;
            }
        }

        scene->AddCollider(static_cast<ColliderType>(colliderType), center, param, velocity);
        engine->AddCollider(static_cast<ColliderType>(colliderType), center, param, velocity);
    }
}

void SceneIO::AppendSimulationParameters(QDomElement root, float timeStep)
{
    QDomElement spNode = m_document.createElement("SimulationParameters");
    
    AppendFloat(spNode, "timeStep", timeStep);
    
    root.appendChild(spNode);
}

void SceneIO::AppendParticleSystem(QDomElement root, Scene* scene)
{
    int count = 0;
    QDomElement pNode = m_document.createElement("ParticleSystem");
    
    for (SceneNodeIterator iter = scene->Begin(); iter.IsValid(); ++iter)
    {
        if ((*iter)->HasRenderable() == true && (*iter)->GetType() == SceneNode::Type::SNOW_CONTAINER)
        {
            QDomElement cNode = m_document.createElement("SnowContainer");
            Mesh* mesh = dynamic_cast<Mesh*>((*iter)->GetRenderable());
            
            AppendString(cNode, "fileName", mesh->GetFileName());
            AppendMatrix(cNode, "CTM", (*iter)->GetCTM());
            
            pNode.appendChild(cNode);
            count++;
        }
    }

    if (count == 0)
    {
        return;
    }

    root.appendChild(pNode);
}

void SceneIO::AppendGrid(QDomElement root, Scene* scene)
{
    SceneNode* gridNode = scene->GetSceneGridNode();
    Grid grid = UISettings::BuildGrid(gridNode->GetCTM());
    QDomElement gNode = m_document.createElement("Grid");
    
    AppendDim(gNode, "gridDim", grid.dim);
    AppendVector(gNode, "pos", grid.pos);
    AppendFloat(gNode, "h", grid.h);
    
    root.appendChild(gNode);
}

//void SceneIO::AppendColliders(QDomElement root, Scene* scene)
//{
//    int count = 0;
//    QDomElement icNode = m_document.createElement("ImplicitColliders");
//    Vector3 velocity;
//
//    for (SceneNodeIterator iter = scene->Begin(); iter.IsValid(); ++iter)
//    {
//        if ((*iter)->hasRenderable() && (*iter)->getType() == SceneNode::SCENE_COLLIDER)
//        {
//            QDomElement cNode = m_document.createElement("Collider");
//            SceneCollider * sCollider = dynamic_cast<SceneCollider*>((*iter)->getRenderable());
//            ImplicitCollider iCollider(*sCollider->getImplicitCollider()); // make copy
//
//            iCollider.applyTransformation((*iter)->getCTM());
//            if (!EQ(sCollider->getVelMag(), 0)) {
//                //                glm::vec4 vel = (*it)->getCTM()*glm::vec4(sCollider->getVelVec(),1.f);
//                //                iCollider.velocity = vec3::normalize(vec3(vel.x,vel.y,vel.z))*sCollider->getVelMag();
//                iCollider.velocity = sCollider->getVelMag() * sCollider->getWorldVelVec((*iter)->getCTM());
//                float mag = sCollider->getVelMag();
//                glm::vec3 velVec = sCollider->getWorldVelVec((*iter)->getCTM());
//            }
//            else iCollider.velocity = vec3(0, 0, 0);
//
//            cNode.setAttribute("type", iCollider.type);
//            appendVector(cNode, "center", iCollider.center);
//            appendVector(cNode, "velocity", iCollider.velocity);
//            appendVector(cNode, "param", iCollider.param);
//            icNode.appendChild(cNode);
//
//            count++;
//        }
//    }
//    if (count > 0)
//        root.appendChild(icNode);
//}
//
//void SceneIO::appendExportSettings(QDomElement root)
//{
//    QDomElement eNode = m_document.createElement("ExportSettings");
//    appendString(eNode, "filePrefix", m_sceneFilePrefix);
//    appendInt(eNode, "maxTime", UiSettings::maxTime());
//    appendInt(eNode, "exportFPS", UiSettings::exportFPS());
//    appendInt(eNode, "exportDensity", UiSettings::exportDensity());
//    appendInt(eNode, "exportVelocity", UiSettings::exportVelocity());
//    root.appendChild(eNode);
//}
//
//void SceneIO::appendString(QDomElement node, const QString name, const QString value)
//{
//    QDomElement sNode = m_document.createElement("string");
//    sNode.setAttribute("name", name);
//    sNode.setAttribute("value", value);
//    node.appendChild(sNode);
//}
//
//void SceneIO::appendInt(QDomElement node, const QString name, const int i)
//{
//    QDomElement iNode = m_document.createElement("int");
//    iNode.setAttribute("name", name);
//    iNode.setAttribute("value", i);
//    node.appendChild(iNode);
//}
//
//void SceneIO::appendFloat(QDomElement node, const QString name, const float f)
//{
//    QDomElement fNode = m_document.createElement("float");
//    fNode.setAttribute("name", name);
//    fNode.setAttribute("value", f);
//    node.appendChild(fNode);
//}
//
//void SceneIO::appendVector(QDomElement node, const QString name, const vec3 v)
//{
//    QDomElement vNode = m_document.createElement("vector");
//    vNode.setAttribute("name", name);
//    vNode.setAttribute("x", v.x);
//    vNode.setAttribute("y", v.y);
//    vNode.setAttribute("z", v.z);
//    node.appendChild(vNode);
//}
//
//void SceneIO::appendDim(QDomElement node, const QString name, const glm::ivec3 iv)
//{
//    QDomElement dNode = m_document.createElement("dim");
//    dNode.setAttribute("name", name);
//    dNode.setAttribute("x", iv.x);
//    dNode.setAttribute("y", iv.y);
//    dNode.setAttribute("z", iv.z);
//    node.appendChild(dNode);
//}
//
//void SceneIO::appendMatrix(QDomElement node, const QString name, const glm::mat4 m)
//{
//    QDomElement mNode = m_document.createElement("matrix");
//    mNode.setAttribute("name", name);
//    QString matstr;
//    matstr.sprintf("%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f",
//        m[0][0], m[1][0], m[2][0], m[3][0],
//        m[0][1], m[1][1], m[2][1], m[3][1],
//        m[0][2], m[1][2], m[2][2], m[3][2],
//        m[0][3], m[1][3], m[2][3], m[3][3]);
//    mNode.setAttribute("value", matstr);
//    node.appendChild(mNode);
//}