/*************************************************************************
> File Name: DataBinding.h
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Data binding classes and functions for various types.
> Created Time: 2017/12/31
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#ifndef SNOW_SIMULATION_DATA_BINDING_H
#define SNOW_SIMULATION_DATA_BINDING_H

#include <Common/Util.h>

#include <QCheckBox>
#include <QComboBox>
#include <QLineEdit>
#include <QObject>
#include <QSlider>
#include <QSpinBox>

class IntBinding : public QObject
{
	Q_OBJECT

public:
	IntBinding(int& value, QObject* parent = nullptr) : QObject(parent), m_value(value)
	{
		// Do nothing
	}

	static IntBinding* BindSpinBox(QSpinBox* spinBox, int& value, QObject* parent = nullptr)
	{
		IntBinding* binding = new IntBinding(value, parent);
		
		spinBox->setValue(value);
		SNOW_ASSERT(connect(spinBox, SIGNAL(valueChanged(int)), binding, SLOT(valueChanged(int))));
		
		return binding;
	}

	static IntBinding* BindLineEdit(QLineEdit* lineEdit, int& value, QObject* parent = nullptr)
	{
		IntBinding* binding = new IntBinding(value, parent);
		
		lineEdit->setText(QString::number(value));
		SNOW_ASSERT(connect(lineEdit, SIGNAL(textChanged(QString)), binding, SLOT(valueChanged(QString))));
		
		return binding;
	}

	static IntBinding* BindSlider(QSlider* slider, int& value, QObject* parent = nullptr)
	{
		IntBinding* binding = new IntBinding(value, parent);
		
		slider->setValue(value);
		SNOW_ASSERT(connect(slider, SIGNAL(valueChanged(int)), binding, SLOT(valueChanged(int))));
		
		return binding;
	}

	static IntBinding* BindTriState(QCheckBox* checkbox, int& value, QObject* parent = nullptr)
	{
		IntBinding* binding = new IntBinding(value, parent);
		
		checkbox->setTristate(true);
		checkbox->setCheckState(static_cast<Qt::CheckState>(value));
		SNOW_ASSERT(connect(checkbox, SIGNAL(stateChanged(int)), binding, SLOT(valueChanged(int))));
		
		return binding;
	}

public slots:
	void valueChanged(int value)
	{
		m_value = value;
	}

	void valueChanged(QString value)
	{
		bool ok = false;
		const int intValue = value.toInt(&ok);
		
		if (ok == true)
		{
			m_value = intValue;
		}
	}

private:
	int& m_value;
};

class FloatBinding : public QObject
{
	Q_OBJECT

public:
	FloatBinding(float& value, QObject* parent = nullptr) : QObject(parent), m_value(value)
	{
		// Do nothing
	}

	static FloatBinding* BindSpinBox(QDoubleSpinBox *spinbox, float& value, QObject* parent = nullptr)
	{
		FloatBinding *binding = new FloatBinding(value, parent);
		
		spinbox->setValue(value);
		SNOW_ASSERT(connect(spinbox, SIGNAL(valueChanged(double)), binding, SLOT(valueChanged(double))));
		
		return binding;
	}

	static FloatBinding* BindLineEdit(QLineEdit* lineEdit, float& value, QObject* parent = nullptr)
	{
		FloatBinding *binding = new FloatBinding(value, parent);
		
		lineEdit->setText(QString::number(value));
		SNOW_ASSERT(connect(lineEdit, SIGNAL(textChanged(QString)), binding, SLOT(valueChanged(QString))));
		
		return binding;
	}

public slots:
	void valueChanged(double value)
	{
		m_value = static_cast<float>(value);
	}

	void valueChanged(QString value)
	{
		bool ok = false;
		const float floatValue = value.toFloat(&ok);

		if (ok == true)
		{
			m_value = floatValue;
		}
	}

private:
	float& m_value;
};

class BoolBinding : public QObject
{
	Q_OBJECT

public:
	BoolBinding(bool& value, QObject* parent = nullptr) : QObject(parent), m_value(value)
	{
		// Do nothing
	}

	static BoolBinding* BindCheckBox(QCheckBox* checkbox, bool& value, QObject* parent = nullptr)
	{
		BoolBinding *binding = new BoolBinding(value, parent);
		
		checkbox->setChecked(value);
		SNOW_ASSERT(connect(checkbox, SIGNAL(toggled(bool)), binding, SLOT(valueChanged(bool))));
		
		return binding;
	}

public slots:
	void valueChanged(bool value)
	{
		m_value = value;
	}

private:
	bool& m_value;
};

class SliderIntAttribute : public QObject
{
	Q_OBJECT

public:
	SliderIntAttribute(QSlider* slider, QLineEdit* edit, int min, int max, int* value, QObject* parent) :
		QObject(parent), m_slider(slider), m_edit(edit), m_value(value)
	{
		if (m_value != nullptr)
		{
			m_slider->setValue(*m_value);
			m_edit->setText(QString::number(*m_value));
		}

		m_slider->setMinimum(min);
		m_slider->setMaximum(max);
		m_edit->setValidator(new QIntValidator(min, max, m_edit));
		
		SNOW_ASSERT(connect(m_slider, SIGNAL(valueChanged(int)), this, SLOT(valueChanged(int))));
		SNOW_ASSERT(connect(m_edit, SIGNAL(textChanged(QString)), this, SLOT(valueChanged(QString))));
	}

	static SliderIntAttribute* BindInt(QSlider* slider, QLineEdit* edit, int min, int max, int* value, QObject* parent)
	{
		return new SliderIntAttribute(slider, edit, min, max, value, parent);
	}

	static SliderIntAttribute* BindSlot(QSlider* slider, QLineEdit* edit, int min, int max, QObject* object, const char* slot)
	{
		SliderIntAttribute* attr = new SliderIntAttribute(slider, edit, min, max, nullptr, object);
		
		SNOW_ASSERT(connect(attr, SIGNAL(attributeChanged(int)), object, slot));
		
		return attr;
	}

	static SliderIntAttribute* bindIntAndSlot(QSlider* slider, QLineEdit* edit, int min, int max, int* value, QObject* object, const char* slot)
	{
		SliderIntAttribute* attr = new SliderIntAttribute(slider, edit, min, max, value, object);
		
		SNOW_ASSERT(connect(attr, SIGNAL(attributeChanged(int)), object, slot));
		
		return attr;
	}

signals:
	void attributeChanged(int value);

public slots:
	void valueChanged(int value)
	{
		bool shouldEmit = true;
		
		if (m_value != nullptr)
		{
			if ((shouldEmit = (*m_value != value)))
			{
				*m_value = value;
			}
		}

		m_slider->setValue(value);
		m_edit->setText(QString::number(value));
		
		if (shouldEmit == true)
		{
			emit attributeChanged(value);
		}
	}

	void valueChanged(QString value)
	{
		valueChanged(value.toInt());
	}

private:
	QSlider* m_slider;
	QLineEdit* m_edit;

	int* m_value;
};

class SliderFloatAttribute : public QObject
{
	Q_OBJECT

public:
	SliderFloatAttribute(QSlider* slider, QLineEdit* edit, float min, float max, float* value, QObject* parent) :
		QObject(parent), m_slider(slider), m_edit(edit), m_value(value), m_min(min), m_max(max)
	{
		if (m_value != nullptr)
		{
			m_slider->setValue(GetIntValue(*m_value));
			m_edit->setText(QString::number(*m_value, 'f', 3));
		}

		m_slider->setMinimum(0);
		m_slider->setMaximum(int(1000 * (m_max - m_min) + 0.5f));
		m_edit->setValidator(new QDoubleValidator(m_min, m_max, 3, m_edit));

		SNOW_ASSERT(connect(m_slider, SIGNAL(valueChanged(int)), this, SLOT(valueChanged(int))));
		SNOW_ASSERT(connect(m_edit, SIGNAL(textChanged(QString)), this, SLOT(valueChanged(QString))));
	}

	static SliderFloatAttribute* BindFloat(QSlider* slider, QLineEdit* edit, float min, float max, float* value, QObject* parent)
	{
		return new SliderFloatAttribute(slider, edit, min, max, value, parent);
	}

	static SliderFloatAttribute* BindSlot(QSlider* slider, QLineEdit* edit, float min, float max, QObject* object, const char* slot)
	{
		SliderFloatAttribute* attr = new SliderFloatAttribute(slider, edit, min, max, nullptr, object);
		
		SNOW_ASSERT(connect(attr, SIGNAL(attributeChanged(float)), object, slot));
		
		return attr;
	}

	static SliderFloatAttribute* BindFloatAndSlot(QSlider* slider, QLineEdit* edit, float min, float max, float* value, QObject* object, const char* slot)
	{
		SliderFloatAttribute* attr = new SliderFloatAttribute(slider, edit, min, max, value, object);
		
		SNOW_ASSERT(connect(attr, SIGNAL(attributeChanged(float)), object, slot));
		
		return attr;
	}

signals:
	void attributeChanged(float value);

public slots:
	void valueChanged(float value)
	{
		bool shouldEmit = true;

		if (m_value != nullptr)
		{
			if ((shouldEmit = (*m_value != value)))
			{
				*m_value = value;
			}
		}

		m_slider->setValue(GetIntValue(value));
		m_edit->setText(QString::number(value, 'f', 3));
		
		if (shouldEmit == true)
		{
			emit attributeChanged(value);
		}
	}

	void valueChanged(int value)
	{
		valueChanged(GetFloatValue(value));
	}

	void valueChanged(QString value)
	{
		valueChanged(value.toFloat());
	}

private:
	QSlider* m_slider;
	QLineEdit* m_edit;

	float* m_value;
	float m_min, m_max;

	inline float GetFloatValue(int i)
	{
		float t = float(i - m_slider->minimum()) / float(m_slider->maximum() - m_slider->minimum());
		return m_min + t * (m_max - m_min);
	}

	inline int GetIntValue(float f)
	{
		float t = (f - m_min) / (m_max - m_min);
		return static_cast<int>(m_slider->minimum() + t * (m_slider->maximum() - m_slider->minimum()) + 0.5f);
	}

};

class CheckboxBoolAttribute : public QObject
{
	Q_OBJECT

public:
	CheckboxBoolAttribute(QCheckBox* checkbox, bool* value, QObject* parent) :
		QObject(parent), m_checkbox(checkbox), m_value(value)
	{
		if (value != nullptr)
		{
			m_checkbox->setChecked(!(*m_value));
			m_checkbox->click();
		}

		SNOW_ASSERT(connect(checkbox, SIGNAL(clicked(bool)), this, SLOT(valueChanged(bool))));
	}

	static CheckboxBoolAttribute* BindBool(QCheckBox* checkbox, bool* value, QObject* parent)
	{
		return new CheckboxBoolAttribute(checkbox, value, parent);
	}

	static CheckboxBoolAttribute* BindSlot(QCheckBox* checkbox, QObject* object, const char* slot)
	{
		CheckboxBoolAttribute* attr = new CheckboxBoolAttribute(checkbox, nullptr, object);
		
		SNOW_ASSERT(connect(attr, SIGNAL(attributedChanged(bool)), object, slot));
		
		return attr;
	}

	static CheckboxBoolAttribute* BindBoolAndSlot(QCheckBox* checkbox, bool* value, QObject* object, const char* slot)
	{
		CheckboxBoolAttribute* attr = new CheckboxBoolAttribute(checkbox, value, object);
		
		SNOW_ASSERT(connect(attr, SIGNAL(attributedChanged(bool)), object, slot));
		
		return attr;
	}

signals:
	void attributedChanged(bool value);

public slots:
	void valueChanged(bool value)
	{
		bool shouldEmit = true;
		
		if (m_value != nullptr)
		{
			if ((shouldEmit = (*m_value != value)))
			{
				*m_value = value;
			}
		}

		m_checkbox->setChecked(value);

		if (shouldEmit)
		{
			emit attributedChanged(value);
		}
	}

private:
	QCheckBox* m_checkbox;

	bool* m_value;
};

class ComboIntAttribute : public QObject
{
	Q_OBJECT

public:
	ComboIntAttribute(QComboBox* combo, int* value, QObject* parent) :
		QObject(parent), m_combo(combo), m_value(value)
	{
		if (m_value != nullptr)
		{
			m_combo->setCurrentIndex(*m_value);
		}

		SNOW_ASSERT(connect(m_combo, SIGNAL(currentIndexChanged(int)), this, SLOT(valueChanged(int))));
	}

	static ComboIntAttribute* BindInt(QComboBox* combo, int* value, QObject* parent)
	{
		return new ComboIntAttribute(combo, value, parent);
	}

	static ComboIntAttribute* BindSlot(QComboBox* combo, QObject* object, const char* slot)
	{
		ComboIntAttribute* attr = new ComboIntAttribute(combo, nullptr, object);
		
		SNOW_ASSERT(connect(attr, SIGNAL(attributeChanged(int)), object, slot));
		
		return attr;
	}

	static ComboIntAttribute* BindIntAndSlot(QComboBox* combo, int* value, QObject* object, const char* slot)
	{
		ComboIntAttribute* attr = new ComboIntAttribute(combo, value, object);
		
		SNOW_ASSERT(connect(attr, SIGNAL(attributeChanged(int)), object, slot));
		
		return attr;
	}

signals:
	void attributeChanged(int value);

public slots:
	void valueChanged(int value)
	{
		bool shouldEmit = true;
		
		if (m_value != nullptr)
		{
			if ((shouldEmit = (*m_value != value)))
			{
				*m_value = value;
			}
		}

		m_combo->setCurrentIndex(value);
	
		if (shouldEmit == true)
		{
			emit attributeChanged(value);
		}
	}

private:
	QComboBox* m_combo;

	int* m_value;
};

#endif