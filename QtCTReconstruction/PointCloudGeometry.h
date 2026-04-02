#pragma once

#include <Qt3DCore/QGeometry>
#include <Qt3DCore/QAttribute>
#include <Qt3DCore/QBuffer>

#include "CTTypes.h"

class PointCloudGeometry : public Qt3DCore::QGeometry {
    Q_OBJECT
public:
    explicit PointCloudGeometry(Qt3DCore::QNode* parent = nullptr);
    ~PointCloudGeometry() override;

    void setPointCloud(const ct::PointCloud& cloud);

private:
    Qt3DCore::QAttribute* m_positionAttribute;
    Qt3DCore::QAttribute* m_colorAttribute;
    Qt3DCore::QBuffer* m_vertexBuffer;
};
