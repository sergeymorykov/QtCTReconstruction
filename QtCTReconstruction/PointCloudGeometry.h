#pragma once

#include <Qt3DCore/QGeometry>
#include <Qt3DCore/QAttribute>
#include <Qt3DCore/QBuffer>
#include <QVector3D>

#include "CTTypes.h"

class PointCloudGeometry : public Qt3DCore::QGeometry {
    Q_OBJECT
    Q_PROPERTY(QVector3D boundsCenter READ boundsCenter NOTIFY boundsChanged)
    Q_PROPERTY(float boundsRadius READ boundsRadius NOTIFY boundsChanged)
    Q_PROPERTY(bool hasData READ hasData NOTIFY boundsChanged)

public:
    explicit PointCloudGeometry(Qt3DCore::QNode* parent = nullptr);
    ~PointCloudGeometry() override;

    void setPointCloud(const ct::PointCloud& cloud);

    QVector3D boundsCenter() const { return m_boundsCenter; }
    float boundsRadius() const { return m_boundsRadius; }
    bool hasData() const { return m_hasData; }

signals:
    void boundsChanged();

private:
    Qt3DCore::QAttribute* m_positionAttribute;
    Qt3DCore::QAttribute* m_colorAttribute;
    Qt3DCore::QBuffer* m_vertexBuffer;

    QVector3D m_boundsCenter = QVector3D(0, 0, 0);
    float m_boundsRadius = 1.0f;
    bool m_hasData = false;
};
