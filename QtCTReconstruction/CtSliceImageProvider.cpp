#include "CtSliceImageProvider.h"

#include "CtReconstructionController.h"

#include <QQmlImageProviderBase>
#include <QImage>

namespace {
bool parseId(const QString& id, QString& kindOut, int& zOut) {
    const auto parts = id.split('/', Qt::KeepEmptyParts);
    if (parts.size() != 2) {
        return false;
    }
    kindOut = parts[0];
    bool ok = false;
    zOut = parts[1].toInt(&ok);
    return ok;
}
} // namespace

CtSliceImageProvider::CtSliceImageProvider(CtReconstructionController* controller)
    : QQuickImageProvider(QQmlImageProviderBase::Image), m_controller(controller) {}

QImage CtSliceImageProvider::requestImage(const QString& id, QSize* size, const QSize& requestedSize) {
    if (m_controller == nullptr) {
        return {};
    }

    QString kind;
    int z = -1;
    if (!parseId(id, kind, z)) {
        return {};
    }

    QImage img;
    if (kind == QStringLiteral("original")) {
        img = m_controller->imageOriginal(z);
    } else if (kind == QStringLiteral("sinogram")) {
        img = m_controller->imageSinogram(z);
    } else if (kind == QStringLiteral("reconstruction")) {
        img = m_controller->imageReconstruction(z);
    } else if (kind == QStringLiteral("difference")) {
        img = m_controller->imageDifference(z);
    } else {
        return {};
    }

    if (img.isNull()) {
        return {};
    }

    QImage out = img;
    if (requestedSize.isValid() && requestedSize.width() > 0 && requestedSize.height() > 0 &&
        (out.size() != requestedSize)) {
        out = img.scaled(requestedSize, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    }

    if (size != nullptr) {
        *size = out.size();
    }

    return out;
}

