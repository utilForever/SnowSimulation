/*************************************************************************
> File Name: CollapsibleBox.h
> Project Name: SnowSimulation
> Author: Chan-Ho Chris Ohk
> Purpose: Collapsible box of snow simulation.
> Created Time: 2018/01/09
> Copyright (c) 2018, Chan-Ho Chris Ohk
*************************************************************************/
#ifndef SNOW_SIMULATION_COLLAPSIBLE_BOX_H
#define SNOW_SIMULATION_COLLAPSIBLE_BOX_H

#include <QGroupBox>

class CollapsibleBox : public QGroupBox
{
	Q_OBJECT
	Q_PROPERTY(bool collapsed READ IsCollapsed WRITE SetCollapsed)

public:
	explicit CollapsibleBox(QWidget* parent);
	~CollapsibleBox();

	bool IsCollapsed() const;

	void Init();

public slots:
	void mousePressEvent(QMouseEvent* event) override;
	void mouseReleaseEvent(QMouseEvent*) override;

	void SetCollapsed(bool collapsed);

protected:
	QString m_rawTitle;

	bool m_clicked;
	bool m_collapsed;

	static void SetWidgetPalette(QWidget* widget, const QPalette& palette);
};

#endif