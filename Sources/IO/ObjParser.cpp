/*************************************************************************
> File Name: ObjParser.cpp
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Obj file parser of snow simulation.
> Created Time: 2018/01/07
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#include <Common/Util.h>
#include <IO/ObjParser.h>

#include <QString>
#include <QStringList>
#include <QRegExp>

void ObjParser::Load(const QString& fileName, QList<Mesh*>& meshes)
{
	ObjParser parser(fileName);

	if (parser.Load() == true)
	{
		while (parser.HasMeshes())
		{
			meshes += parser.PopMesh();
		}
	}
}

bool ObjParser::Save(const QString& fileName, QList<Mesh*>& meshes)
{
	ObjParser parser(fileName);
	parser.SetMeshes(meshes);
	return parser.Save();
}

ObjParser::ObjParser(const QString& fileName) : m_mode(ObjParser::Mode::VERTEX), m_file(fileName), m_currentName("Default")
{
	// Do nothing
}

ObjParser::~ObjParser()
{
	Clear();
}

QString ObjParser::GetFileName() const
{
	return m_file.fileName();
}

void ObjParser::SetFileName(const QString& fileName)
{
	if (m_file.isOpen())
	{
		m_file.close();
	}

	m_file.setFileName(fileName);
}

Mesh* ObjParser::PopMesh()
{
	return m_meshes.dequeue();
}

bool ObjParser::HasMeshes() const
{
	return !m_meshes.empty();
}

void ObjParser::SetMeshes(const QList<Mesh*>& meshes)
{
	m_meshes.clear();
	m_meshes += meshes;
}

bool ObjParser::Load()
{
	Clear();

	if (m_file.fileName().isEmpty() == true)
	{
		LOG("ObjParser: No file name!");
		return false;
	}

	if (m_file.exists() == false || m_file.open(QFile::ReadOnly) == false)
	{
		LOG("ObjParser: Unable to open file %s.", STR(m_file.fileName()));
		return false;
	}

	QString text = QString(m_file.readAll());
	m_file.close();

	QStringList lines = text.split(QRegExp("[\r\n]"), QString::SkipEmptyParts);
	int lineIndex = 0;
	while (lineIndex < lines.size())
	{
		if (Parse(lines, lineIndex) == false)
		{
			return false;
		}
	}

	if (IsMeshPending() == true)
	{
		AddMesh();
	}

	for (QQueue<Mesh*>::iterator iter = m_meshes.begin(); iter != m_meshes.end(); ++iter)
	{
		if ((*iter)->GetNumNormals() != (*iter)->GetNumVertices())
		{
			(*iter)->ComputeNormals();
		}
	}

	return true;
}

bool ObjParser::Save()
{
	if (m_file.fileName().isEmpty() == true)
	{
		LOG("OBJParser: No file name!");
		return false;
	}

	if (m_file.open(QFile::WriteOnly) == false)
	{
		LOG("OBJParser: Unable to open file %s.", STR(m_file.fileName()));
		return false;
	}

	QString string = "";
	while (HasMeshes())
	{
		string += Write(PopMesh());
	}

	m_file.write(string.toLatin1());
	m_file.close();

	return true;
}

void ObjParser::Clear()
{
	while (!m_meshes.empty())
	{
		Mesh* mesh = m_meshes.dequeue();
		
		if (mesh != nullptr)
		{
			delete mesh;
			mesh = nullptr;
		}
	}

	m_vertexPool.clear();
	m_normalPool.clear();
}

bool ObjParser::IsMeshPending() const
{
	return m_vertexPool.empty() == false && m_triPool.empty() == false;
}

void ObjParser::AddMesh()
{
	if (IsMeshPending() == true)
	{
		LOG("ObjParser: Adding mesh %s...", STR(m_currentName));

		Mesh* mesh = new Mesh;
		mesh->SetName(m_currentName);
		mesh->SetFileName(m_file.fileName());
		mesh->SetVertices(m_vertexPool);
		mesh->SetTris(m_triPool);

		if (m_normalPool.size() == m_vertexPool.size())
		{
			mesh->SetNormals(m_normalPool);
		}

		m_meshes.enqueue(mesh);
		m_currentName = "Default";

		m_vertexPool.clear();
		m_triPool.clear();
		m_normalPool.clear();
	}
}

void ObjParser::SetMode(Mode mode)
{
	if (m_mode != mode)
	{
		if (mode == Mode::VERTEX)
		{
			AddMesh();
		}

		m_mode = mode;
	}
}

bool ObjParser::Parse(const QStringList& lines, int& lineIndex)
{
	const QString& line = lines[lineIndex++];
	
	switch (line[0].toLatin1())
	{
	case '#':
		break;
	case 'g':
	case 'o':
		if (ParseName(line) == false)
		{
			LOG("Error parsing name: %s", STR(line));
			return false;
		}
		break;
	case 'v':
		switch (line[1].toLatin1())
		{
		case ' ':
		case '\t':
			if (ParseVertex(line) == false)
			{
				LOG("Error parsing vertex: %s", STR(line));
				return false;
			}
			break;
		default:
			break;
		}
		break;
	case 'f':
		if (ParseFace(line) == false)
		{
			LOG("Error parsing face: %s", STR(line));
			return false;
		}
		break;
	default:
		break;
	}

	return true;
}

bool ObjParser::ParseName(const QString& line)
{
	SetMode(Mode::GROUP);

	const static QRegExp regExp("[\\s+\n\r]");
	QStringList lineStrs = line.split(regExp, QString::SkipEmptyParts);
	m_currentName = (lineStrs.size() > 1) ? lineStrs[1] : "Default";

	return true;
}

bool ObjParser::ParseVertex(const QString& line)
{
	SetMode(Mode::VERTEX);
	
	const static QRegExp regExp("[\\s+v]");
	QStringList vertexStrs = line.split(regExp, QString::SkipEmptyParts);
	bool ok[3];
	m_vertexPool += Vertex(vertexStrs[0].toFloat(&ok[0]), vertexStrs[1].toFloat(&ok[1]), vertexStrs[2].toFloat(&ok[2]));
	
	return ok[0] && ok[1] && ok[2];
}

// Parse face and break into triangles if necessary
bool ObjParser::ParseFace(const QString& line)
{
	SetMode(Mode::FACE);
	
	const static QRegExp regExp("[-\\s+f]");
	QStringList faceStrs = line.split(regExp, QString::SkipEmptyParts);
	int nCorners = faceStrs.size();
	int* indices = new int[nCorners];
	
	for (int i = 0; i < nCorners; ++i)
	{
		const static QRegExp regExp2("[/]");
		QStringList cornerStrs = faceStrs[i].split(regExp2, QString::KeepEmptyParts);
		bool ok;

		// Note: OBJ indices start at 1
		indices[i] = cornerStrs[0].toInt(&ok) - 1;

		if (ok == false)
		{
			return false;
		}
	}

	int nTris = nCorners - 2;
	for (int i = 0; i < nTris; ++i)
	{
		m_triPool += Mesh::Tri(indices[0], indices[i + 1], indices[i + 2]);
	}

	delete[] indices;
	
	return true;
}

QString ObjParser::Write(Mesh* mesh) const
{
	char s[1024];
	QString string = "";

	for (int i = 0; i < mesh->GetNumVertices(); ++i)
	{
		const Vertex& v = mesh->GetVertex(i);
		sprintf_s(s, "v %f %f %f\n", v.x, v.y, v.z);
		string += s;
	}

	string += "g " + mesh->GetName() + "\n";

	for (int i = 0; i < mesh->GetNumTris(); ++i)
	{
		Mesh::Tri t = mesh->GetTri(i);
		// OBJ indices start from 1
		t.Offset(1);
		sprintf_s(s, "f %d %d %d\n", t[0], t[1], t[2]);
		string += s;
	}

	string += "\n";

	return string;
}