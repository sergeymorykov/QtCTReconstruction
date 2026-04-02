#include "CtReconstructionController.h"

#include "FileIO.h"
#include "FilteredBackprojection.h"
#include "Generator3D.h"
#include "RadonTransform.h"
#include "Utils.h"

#include <QCoreApplication>
#include <QDebug>
#include <QFuture>
#include <QImage>
#include <QMutexLocker>
#include <QtConcurrent/QtConcurrentRun>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <memory>
#include <string>

#include <omp.h>

namespace {

std::wstring toWStringBackslashPath(const QString& s) {
    QString t = s;
    t.replace('/', '\\');
    return t.toStdWString();
}

QImage makeEmptyImage() {
    return {};
}

ct::ReconstructionParams defaultReconstructionParams() {
    ct::ReconstructionParams params;
    params.filter = ct::ReconstructionParams::FilterType::SheppLogan;
    params.num_angles = 180;
    params.zero_padding = true;
    return params;
}

} // namespace

CtReconstructionController::CtReconstructionController(QObject* parent)
    : QObject(parent) {}

CtReconstructionController::~CtReconstructionController() = default;

int CtReconstructionController::maxZ() const {
    QMutexLocker lock(&m_mutex);
    return m_maxZ;
}

int CtReconstructionController::currentZ() const {
    QMutexLocker lock(&m_mutex);
    return m_currentZ;
}

bool CtReconstructionController::ready() const {
    QMutexLocker lock(&m_mutex);
    return m_ready;
}

bool CtReconstructionController::running() const {
    QMutexLocker lock(&m_mutex);
    return m_running;
}

bool CtReconstructionController::hasVolume() const {
    QMutexLocker lock(&m_mutex);
    return m_hasVolume;
}

double CtReconstructionController::genTimeSec() const {
    QMutexLocker lock(&m_mutex);
    return m_genTimeSec;
}

double CtReconstructionController::sinogramTimeSec() const {
    QMutexLocker lock(&m_mutex);
    return m_sinogramTimeSec;
}

double CtReconstructionController::reconTimeSec() const {
    QMutexLocker lock(&m_mutex);
    return m_reconTimeSec;
}

void CtReconstructionController::setCurrentZ(const int z) {
    {
        QMutexLocker lock(&m_mutex);
        const int clampedMax = std::max(0, m_maxZ);
        const int clamped = std::min(std::max(0, z), clampedMax);
        if (clamped == m_currentZ) {
            return;
        }
        m_currentZ = clamped;
    }
    emit currentZChanged();
}

QImage CtReconstructionController::getImage(const ImageKind kind, const int z) const {
    QMutexLocker lock(&m_mutex);
    if (!m_ready || z < 0) {
        return makeEmptyImage();
    }

    const size_t zi = static_cast<size_t>(z);
    switch (kind) {
    case ImageKind::Original:
        if (!m_originalImagesPtr) return makeEmptyImage();
        if (zi >= m_originalImagesPtr->size()) return makeEmptyImage();
        return (*m_originalImagesPtr)[zi];
    case ImageKind::Sinogram:
        if (!m_sinogramImagesPtr) return makeEmptyImage();
        if (zi >= m_sinogramImagesPtr->size()) return makeEmptyImage();
        return (*m_sinogramImagesPtr)[zi];
    case ImageKind::Reconstruction:
        if (!m_reconstructionImagesPtr) return makeEmptyImage();
        if (zi >= m_reconstructionImagesPtr->size()) return makeEmptyImage();
        return (*m_reconstructionImagesPtr)[zi];
    case ImageKind::Difference:
        if (!m_differenceImagesPtr) return makeEmptyImage();
        if (zi >= m_differenceImagesPtr->size()) return makeEmptyImage();
        return (*m_differenceImagesPtr)[zi];
    }
    return makeEmptyImage();
}

QImage CtReconstructionController::imageOriginal(const int z) const { return getImage(ImageKind::Original, z); }
QImage CtReconstructionController::imageSinogram(const int z) const { return getImage(ImageKind::Sinogram, z); }
QImage CtReconstructionController::imageReconstruction(const int z) const { return getImage(ImageKind::Reconstruction, z); }
QImage CtReconstructionController::imageDifference(const int z) const { return getImage(ImageKind::Difference, z); }

QImage CtReconstructionController::sliceToImage(const ct::Slice& slice, const bool difference_map) {
    if (slice.empty() || slice[0].empty()) {
        return {};
    }

    const int width = static_cast<int>(slice[0].size());
    const int height = static_cast<int>(slice.size());
    QImage img(width, height, QImage::Format_ARGB32);
    img.fill(0);

    float min_v = slice[0][0];
    float max_v = slice[0][0];
    for (const auto& row : slice) {
        for (const float v : row) {
            min_v = std::min(min_v, v);
            max_v = std::max(max_v, v);
        }
    }
    if (std::abs(max_v - min_v) < 1e-6f) {
        max_v = min_v + 1.0f;
    }

    for (int y = 0; y < height; ++y) {
        auto* rowPtr = reinterpret_cast<QRgb*>(img.scanLine(y));
        for (int x = 0; x < width; ++x) {
            const float v = slice[static_cast<size_t>(height - 1 - y)][static_cast<size_t>(x)];
            if (!difference_map) {
                const float n = ct::utils::clamp((v - min_v) / (max_v - min_v), 0.0f, 1.0f);
                const int c = static_cast<int>(n * 255.0f);
                rowPtr[x] = qRgb(c, c, c);
            } else {
                const float n = ct::utils::clamp(v / 50.0f, -1.0f, 1.0f);
                const int r = static_cast<int>((n > 0.0f) ? n * 255.0f : 0.0f);
                const int b = static_cast<int>((n < 0.0f) ? -n * 255.0f : 0.0f);
                const int g = 255 - std::max(r, b);
                rowPtr[x] = qRgb(r, g, b);
            }
        }
    }

    return img;
}

QImage CtReconstructionController::sinogramToImage(const ct::Sinogram& sinogram) {
    if (sinogram.data.empty() || sinogram.data[0].empty()) {
        return {};
    }

    const int width = static_cast<int>(sinogram.data[0].size());
    const int height = static_cast<int>(sinogram.data.size());
    QImage img(width, height, QImage::Format_ARGB32);
    img.fill(0);

    float min_v = sinogram.data[0][0];
    float max_v = sinogram.data[0][0];
    for (const auto& row : sinogram.data) {
        for (const float v : row) {
            min_v = std::min(min_v, v);
            max_v = std::max(max_v, v);
        }
    }
    if (std::abs(max_v - min_v) < 1e-6f) {
        max_v = min_v + 1.0f;
    }

    for (int y = 0; y < height; ++y) {
        auto* rowPtr = reinterpret_cast<QRgb*>(img.scanLine(y));
        for (int x = 0; x < width; ++x) {
            const float v = sinogram.data[static_cast<size_t>(height - 1 - y)][static_cast<size_t>(x)];
            const float n = ct::utils::clamp((v - min_v) / (max_v - min_v), 0.0f, 1.0f);
            const int c = static_cast<int>(n * 255.0f);
            rowPtr[x] = qRgb(c, c, c);
        }
    }

    return img;
}

void CtReconstructionController::setRunning(const bool running) {
    bool changed = false;
    {
        QMutexLocker lock(&m_mutex);
        if (m_running != running) {
            m_running = running;
            changed = true;
        }
    }
    if (changed) {
        emit runningChanged();
    }
}

void CtReconstructionController::startAsyncTask(QFutureWatcher<ReconstructionResult>* watcher,
                                               const std::function<ReconstructionResult()>& task) {
    if (watcher == nullptr) {
        setRunning(false);
        return;
    }

    connect(watcher, &QFutureWatcher<ReconstructionResult>::finished, this, [this, watcher]() {
        const ReconstructionResult result = watcher ? watcher->result() : ReconstructionResult{};
        applyResult(result);
        if (m_activeWatcher == watcher) {
            m_activeWatcher = nullptr;
        }
        if (watcher) {
            watcher->deleteLater();
        }
    }, Qt::QueuedConnection);

    watcher->setFuture(QtConcurrent::run(task));
}

void CtReconstructionController::generateVolume() {
    QMutexLocker lock(&m_mutex);
    if (m_running) {
        return;
    }
    m_running = true;
    lock.unlock();

    emit runningChanged();

    QPointer<QFutureWatcher<ReconstructionResult>> watcher = new QFutureWatcher<ReconstructionResult>(this);
    m_activeWatcher = watcher;
    // Use helper to start the async task; pass the static generateVolumeTask
    startAsyncTask(watcher, &CtReconstructionController::generateVolumeTask);
}

CtReconstructionController::ReconstructionResult CtReconstructionController::generateVolumeTask() {
    ReconstructionResult result;
    result.success = false;
    result.hasVolume = false;
    result.ready = false;

    const QString appDir = QCoreApplication::applicationDirPath();
    const QString dataDir = appDir + "/data";
    const QString outputDir = dataDir + "/output";
    const std::wstring dataDirW = toWStringBackslashPath(dataDir);
    const std::wstring outputDirW = toWStringBackslashPath(outputDir);

    ct::utils::ensureDirectory(dataDirW);
    ct::utils::ensureDirectory(outputDirW);

    double genTime = 0.0;
    std::string outputDirA = outputDir.toStdString();
    std::replace(outputDirA.begin(), outputDirA.end(), '/', '\\');
    const std::string inputNpyA = outputDirA + "\\synthetic_brain_hu.npy";

    if (static_cast<bool>(std::ifstream(inputNpyA, std::ios::binary))) {
        result.hasVolume = true;
        result.success = true;
    } else {
        const auto t_gen_start = std::chrono::steady_clock::now();
        ct::Generator3D::Params gen_params;
        gen_params.shape = {512, 512, 512};
        gen_params.num_ellipsoids = 300;
        const ct::Volume volume = ct::Generator3D::generateBrainHU(gen_params);
        const std::string outNpyA = outputDirA + "\\synthetic_brain_hu_cxx.npy";
        ct::FileIO::saveVolumeNPY(volume, outNpyA);
        const auto t_gen_end = std::chrono::steady_clock::now();
        genTime = std::chrono::duration<double>(t_gen_end - t_gen_start).count();
        result.hasVolume = true;
        result.success = true;
    }

    result.genTimeSec = genTime;
    return result;
}

void CtReconstructionController::startReconstruction() {
    qDebug() << "[CT] startReconstruction: clicked";
    {
        QMutexLocker lock(&m_mutex);
        qDebug() << "[CT] startReconstruction: state running=" << m_running << "hasVolume=" << m_hasVolume << "ready=" << m_ready;
        if (m_running || !m_hasVolume) {
            qDebug() << "[CT] startReconstruction: ignored";
            return;
        }
        m_running = true;
        m_ready = false;
        m_maxZ = 0;
        m_currentZ = 0;
        m_originalImages.clear();
        m_sinogramImages.clear();
        m_reconstructionImages.clear();
        m_differenceImages.clear();
        m_sinogramTimeSec = 0.0;
        m_reconTimeSec = 0.0;
    }

    emit runningChanged();
    emit readyChanged();
    emit maxZChanged();
    emit currentZChanged();
    emit timingsChanged();

    QPointer<QFutureWatcher<ReconstructionResult>> watcher = new QFutureWatcher<ReconstructionResult>(this);
    m_activeWatcher = watcher;
    // Start reconstruction in background using startAsyncTask to ensure the finished
    // handler runs via a queued connection and won't block the GUI thread.
    startAsyncTask(watcher, &CtReconstructionController::reconstructionTask);
}

CtReconstructionController::ReconstructionResult CtReconstructionController::reconstructionTask() {
    qDebug() << "[CT] computeAll: reconstructionTask start";

    ReconstructionResult results;
    results.success = false;

    const QString inputNpy = QCoreApplication::applicationDirPath() + "/data/output/synthetic_brain_hu_cxx.npy";
    const std::wstring inputNpyW = toWStringBackslashPath(inputNpy);
    const QString outputDirQ = QCoreApplication::applicationDirPath() + "/data/output";
    const std::wstring outputDirW = toWStringBackslashPath(outputDirQ);
    const std::string inputNpyA = inputNpy.toStdString();
    const std::string outputDirA = outputDirQ.toStdString();

    ct::Volume volume;
    if (!ct::FileIO::loadVolumeNPY(inputNpyA, volume, false)) {
        qDebug() << "[CT] computeAll: failed to load volume";
        return results;
    }

    const int depth = static_cast<int>(volume.data.size());
    if (depth <= 0) {
        qDebug() << "[CT] computeAll: empty volume";
        return results;
    }

    results.hasVolume = true;
    results.maxZ = depth - 1;
    results.currentZ = depth / 2;
    results.originalImages = std::make_shared<std::vector<QImage>>(static_cast<size_t>(depth));
    results.sinogramImages = std::make_shared<std::vector<QImage>>(static_cast<size_t>(depth));
    results.reconstructionImages = std::make_shared<std::vector<QImage>>(static_cast<size_t>(depth));
    results.differenceImages = std::make_shared<std::vector<QImage>>(static_cast<size_t>(depth));

    const auto t_sino_start = std::chrono::steady_clock::now();
    ct::ReconstructionParams params = defaultReconstructionParams();
    const int mid_z = depth / 2;
    ct::Slice original_mid;
    ct::Slice recon_mid;
    ct::Slice diff_mid;

    for (int z = 0; z < depth; ++z) {
        if (z == 0 || z == mid_z || z == depth - 1) {
            qDebug() << "[CT] computeAll: sinogram slice" << z << "/" << depth;
        }
        const size_t zi = static_cast<size_t>(z);
        const ct::Slice& original = volume.data[zi];
        ct::Slice normalized = ct::utils::createSlice(original.size(), original[0].size(), 0.0f);

        float hu_min = std::numeric_limits<float>::max();
        float hu_max = std::numeric_limits<float>::lowest();
        for (const auto& row : original) {
            for (const float v : row) {
                hu_min = std::min(hu_min, v);
                hu_max = std::max(hu_max, v);
            }
        }
        if (hu_max > hu_min) {
            const float inv_span = 1.0f / (hu_max - hu_min);
            for (size_t y = 0; y < original.size(); ++y) {
                for (size_t x = 0; x < original[y].size(); ++x) {
                    normalized[y][x] = (original[y][x] - hu_min) * inv_span;
                }
            }
        }

        const size_t detector_bins = original.size();
        ct::Sinogram sinogram = ct::RadonTransform::forward(normalized, params.num_angles, detector_bins);
        (*results.sinogramImages)[zi] = sinogramToImage(sinogram);

        if (z == mid_z) {
            const auto t_sino_end = std::chrono::steady_clock::now();
            results.sinogramTimeSec = std::chrono::duration<double>(t_sino_end - t_sino_start).count();
            qDebug() << "[CT] computeAll: sinogram stage done seconds=" << results.sinogramTimeSec;
        }
    }

    if (results.sinogramTimeSec <= 0.0) {
        const auto t_sino_end = std::chrono::steady_clock::now();
        results.sinogramTimeSec = std::chrono::duration<double>(t_sino_end - t_sino_start).count();
    }

    const auto t_recon_start = std::chrono::steady_clock::now();
    qDebug() << "[CT] computeAll: reconstruction stage start";

    for (int z = 0; z < depth; ++z) {
        if (z == 0 || z == mid_z || z == depth - 1) {
            qDebug() << "[CT] computeAll: reconstruction slice" << z << "/" << depth;
        }

        const size_t zi = static_cast<size_t>(z);
        const ct::Slice& original = volume.data[zi];

        float hu_min = std::numeric_limits<float>::max();
        float hu_max = std::numeric_limits<float>::lowest();
        for (const auto& row : original) {
            for (const float v : row) {
                hu_min = std::min(hu_min, v);
                hu_max = std::max(hu_max, v);
            }
        }

        ct::Slice normalized = ct::utils::createSlice(original.size(), original[0].size(), 0.0f);
        if (hu_max > hu_min) {
            const float inv_span = 1.0f / (hu_max - hu_min);
            for (size_t y = 0; y < original.size(); ++y) {
                for (size_t x = 0; x < original[y].size(); ++x) {
                    normalized[y][x] = (original[y][x] - hu_min) * inv_span;
                }
            }
        }

        const size_t detector_bins = original.size();
        ct::Sinogram sinogram = ct::RadonTransform::forward(normalized, params.num_angles, detector_bins);
        ct::Slice recon_normalized = ct::FilteredBackprojection::reconstruct(sinogram, original.size(), params);

        ct::Slice reconstruction = ct::utils::createSlice(original.size(), original[0].size(), hu_min);
        if (hu_max > hu_min) {
            const float span = (hu_max - hu_min);
            for (size_t y = 0; y < reconstruction.size(); ++y) {
                for (size_t x = 0; x < reconstruction[y].size(); ++x) {
                    reconstruction[y][x] = recon_normalized[y][x] * span + hu_min;
                }
            }
        }

        ct::Slice differences = ct::utils::subtract(reconstruction, original);
        (*results.originalImages)[zi] = sliceToImage(original, false);
        (*results.reconstructionImages)[zi] = sliceToImage(reconstruction, false);
        (*results.differenceImages)[zi] = sliceToImage(differences, true);

        if (z == mid_z) {
            original_mid = original;
            recon_mid = reconstruction;
            diff_mid = differences;
        }
    }

    const auto t_recon_end = std::chrono::steady_clock::now();
    results.reconTimeSec = std::chrono::duration<double>(t_recon_end - t_recon_start).count();

    double mse = 0.0;
    size_t cnt = 0;
    float vmin = std::numeric_limits<float>::max();
    float vmax = std::numeric_limits<float>::lowest();
    for (size_t y = 0; y < diff_mid.size(); ++y) {
        for (size_t x = 0; x < diff_mid[y].size(); ++x) {
            const float d = diff_mid[y][x];
            mse += static_cast<double>(d) * static_cast<double>(d);
            ++cnt;
            vmin = std::min(vmin, original_mid[y][x]);
            vmax = std::max(vmax, original_mid[y][x]);
        }
    }
    mse = (cnt > 0) ? (mse / static_cast<double>(cnt)) : 0.0;
    const double range = static_cast<double>(vmax - vmin);
    const double psnr = (mse <= 0.0 || range <= 0.0) ? std::numeric_limits<double>::infinity() : 10.0 * std::log10((range * range) / mse);

    const std::wstring originalPath = outputDirW + L"\\original_middle.bmp";
    const std::wstring reconstructionPath = outputDirW + L"\\reconstruction_middle.bmp";
    const std::wstring differencePath = outputDirW + L"\\difference_middle.bmp";
    ct::FileIO::saveSliceBMP(original_mid, originalPath, -1000.0f, 100.0f);
    ct::FileIO::saveSliceBMP(recon_mid, reconstructionPath, -1000.0f, 100.0f);
    ct::FileIO::saveSliceBMP(diff_mid, differencePath, -200.0f, 200.0f);

    const std::string metricsPathA = outputDirA + "\\metrics.txt";
    std::ofstream metrics(metricsPathA, std::ios::out | std::ios::trunc);
    if (metrics.is_open()) {
        metrics << std::fixed << std::setprecision(6);
        metrics << "input_npy=" << inputNpyA << "\n";
        metrics << "mid_z=" << mid_z << "\n";
        metrics << "mse=" << mse << "\n";
        metrics << "psnr=";
        if (std::isfinite(psnr)) {
            metrics << psnr;
        } else {
            metrics << "inf";
        }
        metrics << "\n--- Timing (seconds) ---\n";
        metrics << "generation=" << results.genTimeSec << "\n";
        metrics << "sinogram=" << results.sinogramTimeSec << "\n";
        metrics << "reconstruction=" << results.reconTimeSec << "\n";
        metrics << "total=" << (results.genTimeSec + results.sinogramTimeSec + results.reconTimeSec) << "\n";
    }

    results.ready = true;
    results.success = true;
    return results;
}

void CtReconstructionController::applyResult(const ReconstructionResult& result) {
    {
        QMutexLocker lock(&m_mutex);
        m_ready = result.ready;
        m_running = false;
        m_hasVolume = result.hasVolume;
        m_maxZ = result.maxZ;
        m_currentZ = result.currentZ;
        m_genTimeSec = result.genTimeSec;
        m_sinogramTimeSec = result.sinogramTimeSec;
        m_reconTimeSec = result.reconTimeSec;
        m_originalImagesPtr = result.originalImages;
        m_sinogramImagesPtr = result.sinogramImages;
        m_reconstructionImagesPtr = result.reconstructionImages;
        m_differenceImagesPtr = result.differenceImages;
    }

    // Emit signals after releasing the mutex to avoid deadlocks if slots call back into this object.
    emit readyChanged();
    emit runningChanged();
    emit hasVolumeChanged();
    emit maxZChanged();
    emit currentZChanged();
    emit timingsChanged();
}
