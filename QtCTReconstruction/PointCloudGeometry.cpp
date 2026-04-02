#include "PointCloudGeometry.h"
#include <QByteArray>
#include <QColor>
#include <QVector3D>
#include <QtGlobal>
#include <algorithm>
#include <cfloat>

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
        m_hasData = false;
        emit boundsChanged();
        return;
    }

    // === ВЫЧИСЛЕНИЕ BOUNDING BOX ===
    QVector3D minPos(FLT_MAX, FLT_MAX, FLT_MAX);
    QVector3D maxPos(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    for (const auto& p : cloud) {
        QVector3D pos(p.x, p.y, p.z);
        minPos = QVector3D(qMin(minPos.x(), pos.x()),
                          qMin(minPos.y(), pos.y()),
                          qMin(minPos.z(), pos.z()));
        maxPos = QVector3D(qMax(maxPos.x(), pos.x()),
                          qMax(maxPos.y(), pos.y()),
                          qMax(maxPos.z(), pos.z()));
    }

    m_boundsCenter = (minPos + maxPos) * 0.5f;
    // Меняем центр boundsCenter в соответствии с новой ориентацией осей: (x, z, y)
    m_boundsCenter = QVector3D(m_boundsCenter.x(), m_boundsCenter.z(), m_boundsCenter.y());

    QVector3D extent = maxPos - minPos;
    m_boundsRadius = std::max({extent.x(), extent.y(), extent.z()}) * 0.6f;
    m_hasData = true;
    emit boundsChanged();
    // === КОНЕЦ ВЫЧИСЛЕНИЯ ===

    QByteArray bufferData;
    bufferData.resize(cloud.size() * sizeof(VertexData));
    VertexData* vertices = reinterpret_cast<VertexData*>(bufferData.data());

    for (size_t i = 0; i < cloud.size(); ++i) {
        const ct::Point& p = cloud[i];
        // Новая ориентация: Y - это номер среза (высота), Z - глубина (строка изображения)
        vertices[i].position = QVector3D(p.x, p.z, p.y);

        // 8-bit Gray: нормализуем диапазон HU [-1000, 1000] в [0, 1]
        // Воздух (-1000) - черный, Мягкие ткани (0) - серый, Кости (400+) - белый
        float n = std::clamp((p.hu + 1000.0f) / 2000.0f, 0.0f, 1.0f);
        vertices[i].color = QVector3D(n, n, n);
    }

    m_vertexBuffer->setData(bufferData);
    m_positionAttribute->setCount(static_cast<uint>(cloud.size()));
    m_colorAttribute->setCount(static_cast<uint>(cloud.size()));
}
