#include "PointCloudGeometry.h"
#include <QByteArray>
#include <QColor>
#include <QVector3D>
#include <algorithm>

#pragma pack(push, 1)
struct VertexData {
    QVector3D position;
    QVector3D color;
};
#pragma pack(pop)

PointCloudGeometry::PointCloudGeometry(Qt3DCore::QNode* parent)
    : Qt3DCore::QGeometry(parent)
    , m_positionAttribute(new Qt3DCore::QAttribute(this))
    , m_colorAttribute(new Qt3DCore::QAttribute(this))
    , m_vertexBuffer(new Qt3DCore::QBuffer(this))
{
    m_positionAttribute->setName(Qt3DCore::QAttribute::defaultPositionAttributeName());
    m_positionAttribute->setVertexBaseType(Qt3DCore::QAttribute::Float);
    m_positionAttribute->setVertexSize(3);
    m_positionAttribute->setAttributeType(Qt3DCore::QAttribute::VertexAttribute);
    m_positionAttribute->setBuffer(m_vertexBuffer);
    m_positionAttribute->setByteStride(sizeof(VertexData));
    m_positionAttribute->setByteOffset(offsetof(VertexData, position));

    m_colorAttribute->setName(Qt3DCore::QAttribute::defaultColorAttributeName());
    m_colorAttribute->setVertexBaseType(Qt3DCore::QAttribute::Float);
    m_colorAttribute->setVertexSize(3);
    m_colorAttribute->setAttributeType(Qt3DCore::QAttribute::VertexAttribute);
    m_colorAttribute->setBuffer(m_vertexBuffer);
    m_colorAttribute->setByteStride(sizeof(VertexData));
    m_colorAttribute->setByteOffset(offsetof(VertexData, color));

    addAttribute(m_positionAttribute);
    addAttribute(m_colorAttribute);
}

PointCloudGeometry::~PointCloudGeometry() = default;

void PointCloudGeometry::setPointCloud(const ct::PointCloud& cloud) {
    if (cloud.empty()) {
        m_vertexBuffer->setData(QByteArray());
        m_positionAttribute->setCount(0);
        m_colorAttribute->setCount(0);
        return;
    }

    QByteArray bufferData;
    bufferData.resize(cloud.size() * sizeof(VertexData));
    VertexData* vertices = reinterpret_cast<VertexData*>(bufferData.data());

    for (size_t i = 0; i < cloud.size(); ++i) {
        const ct::Point& p = cloud[i];
        vertices[i].position = QVector3D(p.x, p.y, p.z);

        // Расчет цвета (кости: белый, оттенки серого, мягкие ткани: красные/желтые)
        float n = std::clamp((p.hu + 200.0f) / 1200.0f, 0.0f, 1.0f);
        // Простой градиент от красного (мягкие) до белого (кости)
        float r = std::clamp(n + 0.3f, 0.0f, 1.0f);
        float g = std::clamp(n, 0.0f, 1.0f);
        float b = std::clamp(n, 0.0f, 1.0f);
        vertices[i].color = QVector3D(r, g, b);
    }

    m_vertexBuffer->setData(bufferData);
    m_positionAttribute->setCount(static_cast<uint>(cloud.size()));
    m_colorAttribute->setCount(static_cast<uint>(cloud.size()));
}
