#pragma once

#include <QFutureWatcher>
#include <QImage>
#include <QMutex>
#include <QObject>
#include <QPointer>

#include <functional>
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
    Q_PROPERTY(double maxDifference READ maxDifference NOTIFY maxDifferenceChanged)

    Q_PROPERTY(int filterType READ filterType WRITE setFilterType NOTIFY filterTypeChanged)
    Q_PROPERTY(int backendType READ backendType WRITE setBackendType NOTIFY backendTypeChanged)
    Q_PROPERTY(bool asBuffer READ asBuffer WRITE setAsBuffer NOTIFY asBufferChanged)
    Q_PROPERTY(bool isDebugBuild READ isDebugBuild CONSTANT)

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
    double maxDifference() const;

    int filterType() const;
    int backendType() const;
    bool asBuffer() const;
    bool isDebugBuild() const;

    Q_INVOKABLE void generateVolume();
    Q_INVOKABLE void startReconstruction();
    Q_INVOKABLE void savePng(int z);
    Q_INVOKABLE void loadPointCloud();
    Q_INVOKABLE void extractAndFillPointCloud(QObject* geometry);

    void setCurrentZ(int z);
    void setFilterType(int type);
    void setBackendType(int type);
    void setAsBuffer(bool buffer);

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
    void filterTypeChanged();
    void backendTypeChanged();
    void asBufferChanged();
    void maxDifferenceChanged();
    void sliceUpdated(int index);

private:
    enum class ImageKind { Original, Sinogram, Reconstruction, Difference };

public:
    struct ReconstructionResult {
        bool success = false;
        bool hasVolume = false;
        bool ready = false;
        int maxZ = 0;
        int currentZ = 0;
        double genTimeSec = 0.0;
        double sinogramTimeSec = 0.0;
        double reconTimeSec = 0.0;
        double maxDifference = 50.0;
        using ImageVec = std::shared_ptr<std::vector<QImage>>;
        ImageVec originalImages;
        ImageVec sinogramImages;
        ImageVec reconstructionImages;
        ImageVec differenceImages;
    };

private:
    QImage getImage(ImageKind kind, int z) const;

    static QImage sliceToImage(const ct::Slice& slice, bool difference_map, float max_diff = 50.0f);
    static QImage sinogramToImage(const ct::Sinogram& sinogram);

    ReconstructionResult generateVolumeTask();
    
    // Non-static to emit signals
    ReconstructionResult reconstructionTask();
    
    void applyResult(const ReconstructionResult& result);
    void setRunning(bool running);
    void startAsyncTask(QFutureWatcher<ReconstructionResult>* watcher, const std::function<ReconstructionResult()>& task);

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
    double m_maxDifference = 50.0;

    std::vector<QImage> m_originalImages;
    std::vector<QImage> m_sinogramImages;
    std::vector<QImage> m_reconstructionImages;
    std::vector<QImage> m_differenceImages;
    using ImageVec = std::shared_ptr<std::vector<QImage>>;
    ImageVec m_originalImagesPtr;
    ImageVec m_sinogramImagesPtr;
    ImageVec m_reconstructionImagesPtr;
    ImageVec m_differenceImagesPtr;

    int m_filterType = 1; // SheppLogan
    int m_backendType = 0; // OpenCV/CUDA switch
    bool m_asBuffer = true;

    QPointer<QFutureWatcher<ReconstructionResult>> m_activeWatcher;

};
