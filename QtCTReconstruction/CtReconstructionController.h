#pragma once

#include <QImage>
#include <QObject>
#include <QMutex>

#include <vector>

#include "CTTypes.h"

class CtReconstructionController : public QObject {
    Q_OBJECT
    Q_PROPERTY(int maxZ READ maxZ NOTIFY maxZChanged)
    Q_PROPERTY(int currentZ READ currentZ WRITE setCurrentZ NOTIFY currentZChanged)
    Q_PROPERTY(bool ready READ ready NOTIFY readyChanged)
    Q_PROPERTY(bool running READ running NOTIFY runningChanged)
    Q_PROPERTY(bool hasVolume READ hasVolume NOTIFY hasVolumeChanged)
    Q_PROPERTY(double genTimeSec READ genTimeSec NOTIFY timingsChanged)
    Q_PROPERTY(double sinogramTimeSec READ sinogramTimeSec NOTIFY timingsChanged)
    Q_PROPERTY(double reconTimeSec READ reconTimeSec NOTIFY timingsChanged)

public:
    explicit CtReconstructionController(QObject* parent = nullptr);
    ~CtReconstructionController() override;

    int maxZ() const;
    int currentZ() const;
    bool ready() const;
    bool running() const;
    bool hasVolume() const;
    double genTimeSec() const;
    double sinogramTimeSec() const;
    double reconTimeSec() const;

    Q_INVOKABLE void generateVolume();
    Q_INVOKABLE void startReconstruction();

    void setCurrentZ(int z);

    QImage imageOriginal(int z) const;
    QImage imageSinogram(int z) const;
    QImage imageReconstruction(int z) const;
    QImage imageDifference(int z) const;

signals:
    void maxZChanged();
    void currentZChanged();
    void readyChanged();
    void runningChanged();
    void hasVolumeChanged();
    void timingsChanged();

private:
    enum class ImageKind { Original, Sinogram, Reconstruction, Difference };

    QImage getImage(ImageKind kind, int z) const;

    static QImage sliceToImage(const ct::Slice& slice, bool difference_map);
    static QImage sinogramToImage(const ct::Sinogram& sinogram);

    void computeAll();

private:
    mutable QMutex m_mutex;
    bool m_ready = false;
    bool m_running = false;
    bool m_hasVolume = false;

    int m_maxZ = 0;
    int m_currentZ = 0;

    double m_genTimeSec = 0.0;
    double m_sinogramTimeSec = 0.0;
    double m_reconTimeSec = 0.0;

    std::vector<QImage> m_originalImages;
    std::vector<QImage> m_sinogramImages;
    std::vector<QImage> m_reconstructionImages;
    std::vector<QImage> m_differenceImages;

};
