#include "CtReconstructionController.h"

#include "FileIO.h"
#include "FilteredBackprojection.h"
#include "Generator3D.h"
#include "RadonTransform.h"
#include "Utils.h"

#include <QCoreApplication>
#include <QDebug>
#include <QFileDialog>
#include <QFuture>
#include <QGuiApplication>
#include <QImage>
#include <QMutexLocker>
#include <QtConcurrent/QtConcurrentRun>
#include "IReconstructionBackend.h"
#include "PointCloudGeometry.h"

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
#include <cuda_runtime.h>

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
    params.num_angles = 360;
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

double CtReconstructionController::maxDifference() const {
    QMutexLocker lock(&m_mutex);
    return m_maxDifference;
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

int CtReconstructionController::filterType() const { QMutexLocker lock(&m_mutex); return m_filterType; }
int CtReconstructionController::backendType() const { QMutexLocker lock(&m_mutex); return m_backendType; }
bool CtReconstructionController::asBuffer() const { QMutexLocker lock(&m_mutex); return m_asBuffer; }
int CtReconstructionController::volumeSize() const { QMutexLocker lock(&m_mutex); return m_volumeSize; }
bool CtReconstructionController::isDebugBuild() const {
#ifdef NDEBUG
    return false;
#else
    return true;
#endif
}

void CtReconstructionController::setFilterType(int type) {
    {
        QMutexLocker lock(&m_mutex);
        if (m_filterType == type) return;
        m_filterType = type;
    }
    emit filterTypeChanged();
}

void CtReconstructionController::setBackendType(int type) {
    {
        QMutexLocker lock(&m_mutex);
        if (m_backendType == type) return;
        m_backendType = type;
    }
    emit backendTypeChanged();
}

void CtReconstructionController::setAsBuffer(bool buffer) {
    {
        QMutexLocker lock(&m_mutex);
        if (m_asBuffer == buffer) return;
        m_asBuffer = buffer;
    }
    emit asBufferChanged();
}

void CtReconstructionController::setVolumeSize(int size) {
    {
        QMutexLocker lock(&m_mutex);
        if (m_volumeSize == size) return;
        m_volumeSize = size;
        m_hasVolume = false;
        m_ready = false;
        if (m_originalImagesPtr) m_originalImagesPtr->clear();
        if (m_sinogramImagesPtr) m_sinogramImagesPtr->clear();
        if (m_reconstructionImagesPtr) m_reconstructionImagesPtr->clear();
        if (m_differenceImagesPtr) m_differenceImagesPtr->clear();
    }
    emit volumeSizeChanged();
    emit hasVolumeChanged();
    emit readyChanged();
}


void CtReconstructionController::savePng(int z) {
    if (!m_ready) return;

    QString outputDirQ = QCoreApplication::applicationDirPath() + "/data/output/slices";
    QString dir = QFileDialog::getExistingDirectory(nullptr, "Select Output Directory", outputDirQ);
    if (dir.isEmpty()) return;

    int targetZ = (z >= 0) ? z : currentZ();
    if (targetZ < 0 || targetZ > m_maxZ) return;

    QImage img = getImage(ImageKind::Reconstruction, targetZ);
    if (!img.isNull()) {
        QImage target(img.width(), img.height(), QImage::Format_Grayscale8);
        for (int y = 0; y < img.height(); ++y) {
            const QRgb* srcRow = reinterpret_cast<const QRgb*>(img.constScanLine(y));
            uchar* dstRow = target.scanLine(y);
            for (int x = 0; x < img.width(); ++x) {
                dstRow[x] = static_cast<uchar>(qGray(srcRow[x]));
            }
        }
        target.save(dir + QString("/slice_%1.png").arg(targetZ, 4, 10, QChar('0')));
    }
}

void CtReconstructionController::loadPointCloud() {
    qDebug() << "[CT] Load Point Cloud not implemented fully here yet";
}

void CtReconstructionController::extractAndFillPointCloud(QObject* geometry) {
    auto* geom = qobject_cast<PointCloudGeometry*>(geometry);
    if (!geom) {
        qDebug() << "[CT] Cannot cast geometry to PointCloudGeometry";
        return;
    }
    if (!hasVolume()) {
        qDebug() << "[CT] No volume to extract points from";
        return;
    }

    int volSize;
    {
        QMutexLocker lock(&m_mutex);
        volSize = m_volumeSize;
    }
    const QString inputNpy = QCoreApplication::applicationDirPath() + "/data/output/synthetic_brain_hu_cxx_" + QString::number(volSize) + ".npy";
    ct::Volume volume;
    if (!ct::FileIO::loadVolumeNPY(inputNpy.toStdString(), volume, false)) {
        qDebug() << "[CT] Failed to load volume for extraction";
        return;
    }

    int backendId;
    {
        QMutexLocker lock(&m_mutex);
        backendId = m_backendType;
    }

    auto backendImpl = ct::BackendFactory::create(static_cast<ct::BackendFactory::BackendType>(backendId));
    if (!backendImpl || !backendImpl->isAvailable()) {
        backendImpl = ct::BackendFactory::createBestAvailable();
    }

    ct::PointCloud cloud = backendImpl->extractPointCloud(volume, 400.0f);
    geom->setPointCloud(cloud);
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

QImage CtReconstructionController::sliceToImage(const ct::Slice& slice, const bool difference_map, float max_diff) {
    const int width = static_cast<int>(slice.width);
    const int height = static_cast<int>(slice.height);

    // Для разницы используем честный 8-bit Gray формат
    QImage img(width, height, difference_map ? QImage::Format_Grayscale8 : QImage::Format_ARGB32);
    img.fill(0);

    float min_v = slice.empty() ? 0.0f : slice[0][0];
    float max_v = min_v;
    
    #pragma omp parallel for reduction(min:min_v) reduction(max:max_v)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const float v = slice[y][x];
            if (v < min_v) min_v = v;
            if (v > max_v) max_v = v;
        }
    }
    if (std::abs(max_v - min_v) < 1e-6f) {
        max_v = min_v + 1.0f;
    }

    const float span = max_v - min_v;
    const float inv_span = 1.0f / span;
    const int bytes_per_line = img.bytesPerLine();
    uchar* bits = img.bits();

    #pragma omp parallel for
    for (int y = 0; y < height; ++y) {
        uchar* scanLine = bits + y * bytes_per_line;
        for (int x = 0; x < width; ++x) {
            const float v = slice[static_cast<size_t>(height - 1 - y)][static_cast<size_t>(x)];

            float n;
            if (difference_map) {
                // Модуль разницы в диапазоне 0—max_diff
                n = ct::utils::clamp(std::abs(v) / (max_diff > 1e-6f ? max_diff : 1.0f), 0.0f, 1.0f);
            } else {
                // Обычная нормализация для остальных типов изображений
                n = ct::utils::clamp((v - min_v) * inv_span, 0.0f, 1.0f);
            }

            const int c = static_cast<int>(n * 255.0f);

            if (difference_map) {
                scanLine[x] = static_cast<uchar>(c);
            } else {
                reinterpret_cast<QRgb*>(scanLine)[x] = qRgb(c, c, c);
            }
        }
    }

    return img;
}


QImage CtReconstructionController::sinogramToImage(const ct::Sinogram& sinogram) {
    if (sinogram.data.empty()) {
        return {};
    }

    // Fast layout: [Angle][Bin] -> Width = bins, Height = angles
    // For UI: Horizontal = Angles, Vertical = Bins
    const int num_bins = static_cast<int>(sinogram.data.width);
    const int num_angles = static_cast<int>(sinogram.data.height);
    
    // UI Image: Width = Angles, Height = Bins
    QImage img(num_angles, num_bins, QImage::Format_ARGB32);
    img.fill(0);

    float min_v = std::numeric_limits<float>::max();
    float max_v = std::numeric_limits<float>::lowest();
    
    #pragma omp parallel for reduction(min:min_v) reduction(max:max_v)
    for (int a = 0; a < num_angles; ++a) {
        const float* row = sinogram.data[static_cast<size_t>(a)];
        for (int i = 0; i < num_bins; ++i) {
            const float v = row[i];
            if (v < min_v) min_v = v;
            if (v > max_v) max_v = v;
        }
    }
    
    if (std::abs(max_v - min_v) < 1e-6f) {
        max_v = min_v + 1.0f;
    }
    
    const float inv_span = 1.0f / (max_v - min_v);
    const int bytes_per_line = img.bytesPerLine();
    uchar* bits = img.bits();

    // Map [Angle][Bin] to UI Image(x=Angle, y=Bin)
    #pragma omp parallel for
    for (int i = 0; i < num_bins; ++i) {
        uchar* rowPtr = bits + (num_bins - 1 - i) * bytes_per_line;
        QRgb* lineRgb = reinterpret_cast<QRgb*>(rowPtr);
        for (int a = 0; a < num_angles; ++a) {
            const float v = sinogram.data[static_cast<size_t>(a)][static_cast<size_t>(i)];
            const float n = ct::utils::clamp((v - min_v) * inv_span, 0.0f, 1.0f);
            const int c = static_cast<int>(n * 255.0f);
            lineRgb[a] = qRgb(c, c, c);
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
    startAsyncTask(watcher, std::bind(&CtReconstructionController::generateVolumeTask, this));
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

    int backendId;
    int volSize;
    {
        QMutexLocker lock(&m_mutex);
        backendId = m_backendType;
        volSize = m_volumeSize;
    }

    const std::string inputNpyA = outputDirA + "\\synthetic_brain_hu_cxx_" + std::to_string(volSize) + ".npy";

    if (static_cast<bool>(std::ifstream(inputNpyA, std::ios::binary))) {
        result.hasVolume = true;
        result.success = true;
    } else {
        const auto t_gen_start = std::chrono::steady_clock::now();
        ct::Generator3D::Params gen_params;
        gen_params.shape = {static_cast<size_t>(volSize), static_cast<size_t>(volSize), static_cast<size_t>(volSize)}; 
        gen_params.num_ellipsoids = 200;
        
        ct::Generator3D generator;
        if (backendId == 1) { // CUDA
            generator.setBackend(ct::Generator3D::BackendType::CUDA);
            qDebug() << "[CT] Generator using CUDA backend";
        } else {
            generator.setBackend(ct::Generator3D::BackendType::CPU);
            qDebug() << "[CT] Generator using CPU backend";
        }

        const ct::Volume volume = generator.generateBrainHU(gen_params);
        const std::string outNpyA = outputDirA + "\\synthetic_brain_hu_cxx_" + std::to_string(volSize) + ".npy";
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
    startAsyncTask(watcher, std::bind(&CtReconstructionController::reconstructionTask, this));
}

CtReconstructionController::ReconstructionResult CtReconstructionController::reconstructionTask() {
    qDebug() << "[CT] computeAll: reconstructionTask start";

    ReconstructionResult results;
    results.success = false;

    int volSize;
    {
        QMutexLocker lock(&m_mutex);
        volSize = m_volumeSize;
    }

    const QString inputNpy    = QCoreApplication::applicationDirPath() + "/data/output/synthetic_brain_hu_cxx_" + QString::number(volSize) + ".npy";
    const QString outputDirQ  = QCoreApplication::applicationDirPath() + "/data/output";
    const std::wstring outputDirW = toWStringBackslashPath(outputDirQ);
    const std::string inputNpyA  = inputNpy.toStdString();
    const std::string outputDirA = outputDirQ.toStdString();

    ct::Volume volume;
    if (!ct::FileIO::loadVolumeNPY(inputNpyA, volume, false)) {
        qDebug() << "[CT] computeAll: failed to load volume";
        return results;
    }

    // Сохраняем текущее время генерации, чтобы не затереть его нулем в applyResult
    {
        QMutexLocker lock(&m_mutex);
        results.genTimeSec = m_genTimeSec;
    }

    const int depth = static_cast<int>(volume.depth);
    if (depth <= 0) {
        qDebug() << "[CT] computeAll: empty volume";
        return results;
    }

    results.hasVolume = true;
    results.maxZ      = depth - 1;
    results.currentZ  = depth / 2;
    results.originalImages        = std::make_shared<std::vector<QImage>>(static_cast<size_t>(depth));
    results.sinogramImages        = std::make_shared<std::vector<QImage>>(static_cast<size_t>(depth));
    results.reconstructionImages  = std::make_shared<std::vector<QImage>>(static_cast<size_t>(depth));
    results.differenceImages      = std::make_shared<std::vector<QImage>>(static_cast<size_t>(depth));

    // --- Считываем настройки под мьютексом ---
    int backendId;
    int filterId;
    bool asBufferVal;
    {
        QMutexLocker lock(&m_mutex);
        backendId   = m_backendType;
        filterId    = m_filterType;
        asBufferVal = m_asBuffer;
    }

    // --- Параметры реконструкции ---
    ct::ReconstructionParams params = defaultReconstructionParams();
    if      (filterId == 0) params.filter = ct::ReconstructionParams::FilterType::Ramp;
    else if (filterId == 1) params.filter = ct::ReconstructionParams::FilterType::SheppLogan;
    else if (filterId == 2) params.filter = ct::ReconstructionParams::FilterType::Hamming;
    else if (filterId == 3) params.filter = ct::ReconstructionParams::FilterType::Cosine;
    else if (filterId == 4) params.filter = ct::ReconstructionParams::FilterType::Hann;
    else if (filterId == 5) params.filter = ct::ReconstructionParams::FilterType::Bartlett;

    // --- Создаём бэкенд ---
    auto backendImpl = ct::BackendFactory::create(static_cast<ct::BackendFactory::BackendType>(backendId));
    qDebug() << "[CT] reconstructionTask requested backendId:" << backendId;
    if (!backendImpl || !backendImpl->isAvailable()) {
        qDebug() << "[CT] Requested backend not available, falling back to Best Available";
        backendImpl = ct::BackendFactory::createBestAvailable();
    } else {
        qDebug() << "[CT] Requested backend is available!";
    }

    const int mid_z = depth / 2;

    // Уведомляем UI заранее — так он может отображать данные по мере готовности срезов
    {
        QMutexLocker lock(&m_mutex);
        m_originalImagesPtr        = results.originalImages;
        m_sinogramImagesPtr        = results.sinogramImages;
        m_reconstructionImagesPtr  = results.reconstructionImages;
        m_differenceImagesPtr      = results.differenceImages;
        m_maxZ     = results.maxZ;
        m_currentZ = results.currentZ;
        m_ready    = true;
    }
    QMetaObject::invokeMethod(this, "maxZChanged");
    QMetaObject::invokeMethod(this, "currentZChanged");
    QMetaObject::invokeMethod(this, "readyChanged");

    ct::Slice original_mid;
    ct::Slice recon_mid;
    ct::Slice diff_mid;

    // ---------------------------------------------------------------
    // ОПТИМИЗИРОВАНО: Пакетная реконструкция всего объема сразу.
    // Это устраняет пробелы в работе GPU и накладные расходы API.
    // ---------------------------------------------------------------
    const auto t_total_start = std::chrono::steady_clock::now();

    qDebug() << "[CT] computeAll: starting batch volume reconstruction on backend...";
    
    ct::Volume reconstructed_volume;

    // --- ТОЧНОЕ ПРОФИЛИРОВАНИЕ GPU ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    try {
        backendImpl->reconstructVolume(volume, reconstructed_volume, params, nullptr);
    } catch (const std::exception& e) {
        qDebug() << "[CT] EXCEPTION in reconstructVolume:" << e.what();
        results.success = false;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return results;
    }
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float gpu_compute_ms = 0;
    cudaEventElapsedTime(&gpu_compute_ms, start, stop);

    const auto t_recon_done = std::chrono::steady_clock::now();
    results.sinogramTimeSec = backendImpl->lastSinogramTimeMs() / 1000.0;

    // Используем backend-репортируемое время реконструкции, если оно есть
    // (HybridBackend измеряет ЭТАП II отдельно). Иначе fallback на wall-clock
    // (CUDABackend и др. не разделяют фазы, для них lastReconstructionTimeMs()=0).
    const double backend_recon_ms = backendImpl->lastReconstructionTimeMs();
    if (backend_recon_ms > 0.0) {
        results.reconTimeSec = backend_recon_ms / 1000.0;
    } else {
        results.reconTimeSec = std::chrono::duration<double>(t_recon_done - t_total_start).count()
                              - results.sinogramTimeSec;
    }

    qDebug() << "[CT] computeAll: batch recon done.";
    qDebug() << "[BENCH] GPU Compute Only:" << gpu_compute_ms << "ms";
    qDebug() << "[BENCH] Total (with CPU logic):" << (results.reconTimeSec * 1000.0) << "ms";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    qDebug() << "[CT] computeAll: Generating images for UI...";

    double global_mse = 0.0;
    double global_mae = 0.0;
    float max_abs = 0.0f;
    size_t cnt = 0;
    float vmin = std::numeric_limits<float>::max();
    float vmax = std::numeric_limits<float>::lowest();

    // Первый проход: метрики
    #pragma omp parallel for reduction(+:global_mse, global_mae, cnt) reduction(max:max_abs, vmax) reduction(min:vmin)
    for (int z = 0; z < depth; ++z) {
        const size_t zi = static_cast<size_t>(z);
        const ct::Slice original = volume.getSlice(zi);
        const ct::Slice reconstruction = reconstructed_volume.getSlice(zi);

        for (size_t y = 0; y < original.height; ++y) {
            for (size_t x = 0; x < original.width; ++x) {
                const float diff = reconstruction[y][x] - original[y][x];
                const float orig = original[y][x];
                
                global_mse += static_cast<double>(diff) * static_cast<double>(diff);
                global_mae += std::abs(static_cast<double>(diff));
                max_abs = std::max(max_abs, std::abs(diff));
                vmin = std::min(vmin, orig);
                vmax = std::max(vmax, orig);
                cnt++;
            }
        }
    }

    if (max_abs < 1e-6f) max_abs = 1.0f;
    results.maxDifference = max_abs;

    // Второй проход: генерация QImage
    #pragma omp parallel for
    for (int z = 0; z < depth; ++z) {
        const size_t zi = static_cast<size_t>(z);
        const ct::Slice original = volume.getSlice(zi);
        const ct::Slice reconstruction = reconstructed_volume.getSlice(zi);
        ct::Slice differences = ct::utils::subtract(reconstruction, original);

        if (results.sinogramImages && zi < results.sinogramImages->size()) {
            ct::Sinogram temp_sino = backendImpl->computeSinogram(original, params.num_angles, original.width);
            (*results.sinogramImages)[zi] = sinogramToImage(temp_sino);
        }

        (*results.originalImages)[zi]       = sliceToImage(original, false);
        (*results.reconstructionImages)[zi] = sliceToImage(reconstruction, false);
        (*results.differenceImages)[zi]     = sliceToImage(differences, true, max_abs);

        if (!asBufferVal && (z % 10 == 0 || z == depth - 1)) {
            QMetaObject::invokeMethod(this, [this, z]() { emit sliceUpdated(z); }, Qt::QueuedConnection);
        }

        if (z == mid_z) {
            original_mid = original;
            recon_mid    = reconstruction;
            diff_mid     = differences;
        }
    }

    const auto t_total_end = std::chrono::steady_clock::now();
    const double totalSec = std::chrono::duration<double>(t_total_end - t_total_start).count();
    // НЕ переписываем results.reconTimeSec: оно уже зафиксировано выше, ДО UI-loop'а.
    // Постобработка (повторный computeSinogram для UI, генерация QImage) НЕ должна
    // засчитываться как «время реконструкции» — иначе backend несправедливо обвиняется.
    (void)totalSec;

    global_mse = (cnt > 0) ? (global_mse / static_cast<double>(cnt)) : 0.0;
    global_mae = (cnt > 0) ? (global_mae / static_cast<double>(cnt)) : 0.0;

    const double range = static_cast<double>(vmax - vmin);
    const double psnr  = (global_mse <= 0.0 || range <= 0.0)
                         ? std::numeric_limits<double>::infinity()
                         : 10.0 * std::log10((range * range) / global_mse);

    qDebug() << "[METRICS] Global Max Absolute Error =" << max_abs;
    qDebug() << "[METRICS] Global MAE =" << global_mae;
    qDebug() << "[METRICS] Global MSE =" << global_mse;
    qDebug() << "[METRICS] Global PSNR =" << psnr;

    // --- BMP для анализа ---
    const std::wstring originalPath       = outputDirW + L"\\original_middle.bmp";
    const std::wstring reconstructionPath = outputDirW + L"\\reconstruction_middle.bmp";
    const std::wstring differencePath     = outputDirW + L"\\difference_middle.bmp";
    ct::FileIO::saveSliceBMP(original_mid, originalPath,       -1000.0f, 100.0f);
    ct::FileIO::saveSliceBMP(recon_mid,    reconstructionPath, -1000.0f, 100.0f);
    ct::FileIO::saveSliceBMP(diff_mid,     differencePath,     -200.0f,  200.0f);

    // --- metrics.txt ---
    const std::string metricsPathA = outputDirA + "\\metrics.txt";
    std::ofstream metrics(metricsPathA, std::ios::out | std::ios::trunc);
    if (metrics.is_open()) {
        metrics << std::fixed << std::setprecision(6);
        metrics << "input_npy=" << inputNpyA << "\n";
        metrics << "mid_z=" << mid_z << "\n";
        metrics << "global_mse=" << global_mse << "\n";
        metrics << "global_mae=" << global_mae << "\n";
        metrics << "max_abs_error=" << max_abs << "\n";
        metrics << "psnr=";
        if (std::isfinite(psnr)) { metrics << psnr; } else { metrics << "inf"; }
        metrics << "\n--- Timing (seconds) ---\n";
        metrics << "generation="     << results.genTimeSec      << "\n";
        metrics << "sinogram="       << results.sinogramTimeSec << "\n";
        metrics << "reconstruction=" << results.reconTimeSec    << "\n";
        metrics << "total="          << (results.genTimeSec + results.sinogramTimeSec + results.reconTimeSec) << "\n";
    }

    results.ready   = true;
    results.success = true;
    return results;
}

void CtReconstructionController::applyResult(const ReconstructionResult& result) {
    {
        QMutexLocker lock(&m_mutex);
        m_ready       = result.ready;
        m_running     = false;
        m_hasVolume   = result.hasVolume;
        m_maxZ        = result.maxZ;
        m_currentZ    = result.currentZ;
        if (result.genTimeSec > 0) {
            m_genTimeSec = result.genTimeSec;
        }
        m_sinogramTimeSec  = result.sinogramTimeSec;
        m_reconTimeSec     = result.reconTimeSec;
        m_maxDifference    = result.maxDifference;
        m_originalImagesPtr        = result.originalImages;
        m_sinogramImagesPtr        = result.sinogramImages;
        m_reconstructionImagesPtr  = result.reconstructionImages;
        m_differenceImagesPtr      = result.differenceImages;
    }

    // Сигналы испускаем после снятия мьютекса, чтобы избежать дедлоков
    emit readyChanged();
    emit runningChanged();
    emit hasVolumeChanged();
    emit maxZChanged();
    emit currentZChanged();
    emit timingsChanged();
    emit maxDifferenceChanged();
}
