#pragma once

#include <QImage>
#include <QQuickImageProvider>
#include <QSize>

class CtReconstructionController;

class CtSliceImageProvider : public QQuickImageProvider {
public:
    explicit CtSliceImageProvider(CtReconstructionController* controller);
    ~CtSliceImageProvider() override = default;

    QImage requestImage(const QString& id, QSize* size, const QSize& requestedSize) override;

private:
    CtReconstructionController* m_controller = nullptr; // owned by QObject parent chain
};

