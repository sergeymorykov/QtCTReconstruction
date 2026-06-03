#include "CtReconstructionController.h"

#include "FileIO.h"
#include "FilteredBackprojection.h"
#include "Generator3D.h"
#include "RadonTransform.h"
#include "Utils.h"

#include <QCoreApplication>
#include <QDebug>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFuture>
#include <QGuiApplication>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QMutexLocker>
#include <QRegularExpression>
#include <QSettings>
#include <QStringList>
#include <QTextStream>
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


// Конвертирует ARGB QImage в Grayscale8 PNG-готовый QImage (как раньше делал savePng).
// Вынесено, чтобы не дублировать в одиночном и пакетном путях экспорта.
static QImage toGrayscalePngImage(const QImage& src) {
    QImage target(src.width(), src.height(), QImage::Format_Grayscale8);
    for (int y = 0; y < src.height(); ++y) {
        const QRgb* srcRow = reinterpret_cast<const QRgb*>(src.constScanLine(y));
        uchar* dstRow = target.scanLine(y);
        for (int x = 0; x < src.width(); ++x) {
            dstRow[x] = static_cast<uchar>(qGray(srcRow[x]));
        }
    }
    return target;
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
        toGrayscalePngImage(img).save(dir + QString("/slice_%1.png").arg(targetZ, 4, 10, QChar('0')));
    }
}

void CtReconstructionController::exportAllReconstructionPng() {
    if (!m_ready) {
        qDebug() << "[CT] exportAllReconstructionPng: reconstruction not ready";
        return;
    }

    QString outputDirQ = QCoreApplication::applicationDirPath() + "/data/output/slices";
    QString dir = QFileDialog::getExistingDirectory(nullptr, "Select Output Directory for all slices", outputDirQ);
    if (dir.isEmpty()) return;

    int totalZ;
    {
        QMutexLocker lock(&m_mutex);
        totalZ = m_maxZ + 1;
        if (!m_reconstructionImagesPtr || m_reconstructionImagesPtr->empty()) {
            qDebug() << "[CT] exportAllReconstructionPng: no reconstruction images";
            return;
        }
    }

    const auto t0 = std::chrono::steady_clock::now();
    int written = 0;
    for (int z = 0; z < totalZ; ++z) {
        QImage img = getImage(ImageKind::Reconstruction, z);
        if (img.isNull()) continue;
        const QString path = dir + QString("/slice_%1.png").arg(z, 4, 10, QChar('0'));
        if (toGrayscalePngImage(img).save(path)) {
            ++written;
        }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double secs = std::chrono::duration<double>(t1 - t0).count();
    qDebug() << "[CT] exportAllReconstructionPng: saved" << written << "of" << totalZ
             << "slices in" << secs << "s to" << dir;
}

// ============================================================
//  Point cloud loading: NPY / PLY (ASCII) / CSV
// ============================================================

// PLY ASCII парсер: ищет element vertex N, property float x/y/z + одно скалярное
// свойство для HU (intensity / scalar / value / red — берётся первое подходящее).
// Поддерживает только ASCII PLY с float-свойствами; binary не поддерживается.
static bool loadPointCloudPLY(const QString& path, ct::PointCloud& out) {
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) return false;
    QTextStream ts(&f);

    QString line = ts.readLine().trimmed();
    if (line != "ply") return false;
    line = ts.readLine().trimmed();
    if (!line.startsWith("format ascii")) {
        qDebug() << "[CT] PLY loader: only ascii format supported, got:" << line;
        return false;
    }

    int vertexCount = 0;
    QStringList props;
    bool inVertex = false;
    while (!ts.atEnd()) {
        line = ts.readLine().trimmed();
        if (line.startsWith("comment")) continue;
        if (line.startsWith("element vertex")) {
            vertexCount = line.section(' ', 2, 2).toInt();
            inVertex = true;
            continue;
        }
        if (line.startsWith("element ")) {
            inVertex = false;
            continue;
        }
        if (inVertex && line.startsWith("property ")) {
            props.append(line.section(' ', -1, -1));  // property name
        }
        if (line == "end_header") break;
    }
    if (vertexCount <= 0) return false;

    const int ix = props.indexOf("x");
    const int iy = props.indexOf("y");
    const int iz = props.indexOf("z");
    if (ix < 0 || iy < 0 || iz < 0) return false;
    int ihu = -1;
    for (const auto& name : { QString("hu"), QString("intensity"), QString("scalar"),
                              QString("value"), QString("density"), QString("red") }) {
        ihu = props.indexOf(name);
        if (ihu >= 0) break;
    }

    out.clear();
    out.reserve(vertexCount);
    const int totalProps = props.size();
    for (int i = 0; i < vertexCount && !ts.atEnd(); ++i) {
        const QStringList toks = ts.readLine().split(QRegularExpression("\\s+"), Qt::SkipEmptyParts);
        if (toks.size() < totalProps) continue;
        ct::Point p;
        p.x  = toks[ix].toFloat();
        p.y  = toks[iy].toFloat();
        p.z  = toks[iz].toFloat();
        p.hu = (ihu >= 0) ? toks[ihu].toFloat() : 0.0f;
        out.push_back(p);
    }
    return !out.empty();
}

// CSV парсер: каждая строка "x,y,z[,hu]" (разделитель — запятая или ;).
// Если первая строка не парсится как числа — считаем её header и пропускаем.
static bool loadPointCloudCSV(const QString& path, ct::PointCloud& out) {
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) return false;
    QTextStream ts(&f);

    out.clear();
    bool firstLine = true;
    while (!ts.atEnd()) {
        QString line = ts.readLine().trimmed();
        if (line.isEmpty()) continue;
        const QStringList toks = line.split(QRegularExpression("[,;\\s]+"), Qt::SkipEmptyParts);
        if (toks.size() < 3) continue;
        bool okX = false, okY = false, okZ = false;
        const float x = toks[0].toFloat(&okX);
        const float y = toks[1].toFloat(&okY);
        const float z = toks[2].toFloat(&okZ);
        if (!okX || !okY || !okZ) {
            if (firstLine) { firstLine = false; continue; }  // header
            continue;
        }
        firstLine = false;
        ct::Point p;
        p.x = x; p.y = y; p.z = z;
        p.hu = (toks.size() >= 4) ? toks[3].toFloat() : 0.0f;
        out.push_back(p);
    }
    return !out.empty();
}

// Растеризация разреженного облака точек в плотный воксельный том N_tgt³.
//
// Старая версия маппила каждую точку в один воксель target-сетки через
// bbox-нормализацию. При N_tgt > N_source оставались зазоры в регулярных
// позициях → moire/чёрный экран на 512³/1024³. При N_tgt < N_source наоборот,
// несколько точек давили друг друга в одном вокселе.
//
// Новый алгоритм — двухпроходный resample:
//   1. По bbox облака восстанавливаем размер ИСХОДНОЙ сетки N_src
//      (для облаков от np.where(...) это размер MRI/CT-сетки, типично 192-256).
//   2. Растеризуем точки в плотный том N_src³ — каждый source-воксель получает
//      своё HU-значение, пустые остаются 0 (это были "фон/воздух", отсечённые
//      порогом в Python-скрипте).
//   3. Resample N_src³ → N_tgt³ через nearest-neighbor (по центрам вокселей).
//      Upsample = replication (каждый source-воксель копируется в k³ target,
//      где k = N_tgt/N_src), нет дырок. Downsample = decimation.
//
// Память: source-буфер = N_src³ × 4 байт. Для типичной MRI 256³ это 64 МБ,
// 512³ — 512 МБ. Если облако имеет очень большой bbox — может занять много.
static void rasterizeCloudToVolume(const ct::PointCloud& cloud, int N_tgt, ct::Volume& out) {
    out.assign((size_t)N_tgt, (size_t)N_tgt, (size_t)N_tgt, 0.0f);
    if (cloud.empty() || N_tgt <= 0) return;

    // 1. Bbox облака.
    float xmin = cloud[0].x, xmax = xmin;
    float ymin = cloud[0].y, ymax = ymin;
    float zmin = cloud[0].z, zmax = zmin;
    for (const auto& p : cloud) {
        xmin = std::min(xmin, p.x); xmax = std::max(xmax, p.x);
        ymin = std::min(ymin, p.y); ymax = std::max(ymax, p.y);
        zmin = std::min(zmin, p.z); zmax = std::max(zmax, p.z);
    }
    const float xrange = xmax - xmin;
    const float yrange = ymax - ymin;
    const float zrange = zmax - zmin;
    const float maxRange = std::max({xrange, yrange, zrange});

    // 2. Восстановление исходного размера сетки. Для облаков от np.where(...)
    //    координаты — целые индексы, и max + 1 даёт точный размер MRI-сетки.
    //    Для произвольных координат — округляем bbox в большую сторону.
    int N_src = std::max(4, (int)std::round(maxRange) + 1);
    // Ограничение по памяти: 1024³ × 4 = 4 ГБ. Для облаков с очень большим
    // bbox обрезаем N_src — это понизит детализацию, но не свалит процесс.
    constexpr int N_SRC_LIMIT = 1024;
    if (N_src > N_SRC_LIMIT) {
        qDebug() << "[CT] rasterizeCloudToVolume: clamping N_src" << N_src
                 << "→" << N_SRC_LIMIT << "(bbox too large)";
        N_src = N_SRC_LIMIT;
    }
    qDebug() << "[CT] rasterizeCloudToVolume: bbox" << xrange << "×" << yrange << "×" << zrange
             << "→ N_src=" << N_src << ", N_tgt=" << N_tgt;

    // 3. Растеризация в плотный source-том. Координаты bbox-relative,
    //    масштаб точек в [0, N_src-1].
    const size_t srcSize = (size_t)N_src * N_src * N_src;
    std::vector<float> src;
    try {
        src.assign(srcSize, 0.0f);
    } catch (const std::bad_alloc&) {
        qDebug() << "[CT] rasterizeCloudToVolume: bad_alloc for source" << N_src
                 << "³ (~ " << (srcSize * 4) / (1024 * 1024) << "MB)";
        return;
    }

    const float srcScale = (maxRange > 1e-6f) ? ((float)(N_src - 1) / maxRange) : 1.0f;
    for (const auto& p : cloud) {
        int sx = (int)((p.x - xmin) * srcScale + 0.5f);
        int sy = (int)((p.y - ymin) * srcScale + 0.5f);
        int sz = (int)((p.z - zmin) * srcScale + 0.5f);
        if (sx < 0) sx = 0; else if (sx >= N_src) sx = N_src - 1;
        if (sy < 0) sy = 0; else if (sy >= N_src) sy = N_src - 1;
        if (sz < 0) sz = 0; else if (sz >= N_src) sz = N_src - 1;
        src[((size_t)sz * N_src + sy) * N_src + sx] = p.hu;
    }

    // 4. Resample source → target через nearest-neighbor по центрам вокселей.
    //    Маппинг: tx = центр target-вокселя в [0, N_tgt) → s = (tx + 0.5) * N_src / N_tgt.
    //    Для N_tgt > N_src: каждый source-воксель копируется ⌈N_tgt/N_src⌉^3 раз
    //    в target — нет дырок (которые давали moire в старом варианте).
    const float ratio = (float)N_src / (float)N_tgt;
    #pragma omp parallel for
    for (int tz = 0; tz < N_tgt; ++tz) {
        int sz = (int)((tz + 0.5f) * ratio);
        if (sz < 0) sz = 0; else if (sz >= N_src) sz = N_src - 1;
        for (int ty = 0; ty < N_tgt; ++ty) {
            int sy = (int)((ty + 0.5f) * ratio);
            if (sy < 0) sy = 0; else if (sy >= N_src) sy = N_src - 1;
            for (int tx = 0; tx < N_tgt; ++tx) {
                int sx = (int)((tx + 0.5f) * ratio);
                if (sx < 0) sx = 0; else if (sx >= N_src) sx = N_src - 1;
                out.data[((size_t)tz * N_tgt + ty) * N_tgt + tx] =
                    src[((size_t)sz * N_src + sy) * N_src + sx];
            }
        }
    }
}

bool CtReconstructionController::loadPointCloud() {
    const QString defaultDir = QCoreApplication::applicationDirPath() + "/data/output";
    const QString path = QFileDialog::getOpenFileName(
        nullptr, "Load Point Cloud or Volume", defaultDir,
        "Point cloud / Volume NPY (*.npy *.ply *.csv);;NumPy (*.npy);;PLY ASCII (*.ply);;CSV (*.csv);;All files (*)");
    if (path.isEmpty()) return false;

    // ============================================================
    // Распознаём формат: dense volume NPY (shape Z,Y,X) или sparse
    // point cloud (NPY shape N×4, PLY, CSV).
    //
    // Volume path: данные кладутся прямо в стандартный NPY, который
    //   reconstructionTask читает. m_loadedCloud остаётся пустым →
    //   QML не открывает 3D viewer (нечего показывать).
    //
    // Cloud path: точки сохраняются в m_loadedCloud (для 3D viewer'а)
    //   и растеризуются в том через rasterizeCloudToVolume.
    // ============================================================
    ct::PointCloud cloud;
    ct::Volume volume;
    bool gotCloud  = false;
    bool gotVolume = false;
    const QString lower = path.toLower();

    if (lower.endsWith(".npy")) {
        // Сначала пробуем как point cloud (быстрая проверка по header'у).
        if (ct::FileIO::loadPointCloudNPY(path.toStdString(), cloud)) {
            gotCloud = true;
        } else if (ct::FileIO::loadVolumeNPY(path.toStdString(), volume, /*input_is_xyz_layout=*/false)) {
            gotVolume = true;
        }
    } else if (lower.endsWith(".ply")) {
        gotCloud = loadPointCloudPLY(path, cloud);
    } else if (lower.endsWith(".csv")) {
        gotCloud = loadPointCloudCSV(path, cloud);
    } else {
        // Без явного расширения — пробуем все парсеры по очереди.
        if (ct::FileIO::loadPointCloudNPY(path.toStdString(), cloud)) gotCloud = true;
        else if (ct::FileIO::loadVolumeNPY(path.toStdString(), volume, false)) gotVolume = true;
        else if (loadPointCloudPLY(path, cloud)) gotCloud = true;
        else if (loadPointCloudCSV(path, cloud)) gotCloud = true;
    }

    if (!gotCloud && !gotVolume) {
        qDebug() << "[CT] loadPointCloud: failed to parse" << path;
        return false;
    }
    if (gotCloud && cloud.empty()) {
        qDebug() << "[CT] loadPointCloud: empty cloud in" << path;
        return false;
    }

    // ensureDirectory — НЕ рекурсивный, создаём data и data/output отдельно.
    const QString appDir   = QCoreApplication::applicationDirPath();
    const QString dataDir  = appDir + "/data";
    const QString outDir   = dataDir + "/output";
    ct::utils::ensureDirectory(toWStringBackslashPath(dataDir));
    ct::utils::ensureDirectory(toWStringBackslashPath(outDir));

    int volSize = 0;
    const auto t0 = std::chrono::steady_clock::now();

    if (gotVolume) {
        // ---------- VOLUME PATH ----------
        // Подбираем m_volumeSize под загруженный том. Если том не кубический,
        // используем max-измерение и пишем предупреждение (FBP в апп ждёт куб).
        const size_t w = volume.width, h = volume.height, d = volume.depth;
        if (w != h || h != d) {
            qDebug() << "[CT] loadPointCloud: WARNING — volume is non-cubic"
                     << w << "×" << h << "×" << d << "; FBP expects cube, taking max dim";
        }
        const size_t N = std::max({w, h, d});
        volSize = (int)N;

        {
            QMutexLocker lock(&m_mutex);
            m_loadedCloud.clear();   // нет облака — 3D viewer не открываем
            m_volumeSize = volSize;
        }
        emit volumeSizeChanged();
        qDebug() << "[CT] loadPointCloud: loaded dense volume" << w << "×" << h << "×" << d
                 << "from" << path << "; saving as standard NPY for reconstruction";

        const QString npyPath = outDir + "/synthetic_brain_hu_cxx_" + QString::number(volSize) + ".npy";
        if (!ct::FileIO::saveVolumeNPY(volume, npyPath.toStdString())) {
            qDebug() << "[CT] loadPointCloud: failed to save volume to" << npyPath;
            return false;
        }
    } else {
        // ---------- CLOUD PATH ----------
        {
            QMutexLocker lock(&m_mutex);
            m_loadedCloud = std::move(cloud);
            volSize = m_volumeSize;
        }
        qDebug() << "[CT] loadPointCloud: loaded" << m_loadedCloud.size() << "points from" << path;

        ct::Volume vol;
        rasterizeCloudToVolume(m_loadedCloud, volSize, vol);

        const QString npyPath = outDir + "/synthetic_brain_hu_cxx_" + QString::number(volSize) + ".npy";
        if (!ct::FileIO::saveVolumeNPY(vol, npyPath.toStdString())) {
            qDebug() << "[CT] loadPointCloud: failed to save rasterized volume to" << npyPath;
            // Облако загружено — 3D viewer откроется; реконструкция не активна.
            return true;
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    qDebug() << "[CT] loadPointCloud: prepared volume of size" << volSize << "³ in"
             << std::chrono::duration<double>(t1 - t0).count() << "s";

    {
        QMutexLocker lock(&m_mutex);
        m_hasVolume = true;
        m_ready = false;   // предыдущая реконструкция (если была) больше не валидна
        if (m_originalImagesPtr) m_originalImagesPtr->clear();
        if (m_sinogramImagesPtr) m_sinogramImagesPtr->clear();
        if (m_reconstructionImagesPtr) m_reconstructionImagesPtr->clear();
        if (m_differenceImagesPtr) m_differenceImagesPtr->clear();
    }
    emit hasVolumeChanged();
    emit readyChanged();
    return true;
}

bool CtReconstructionController::hasLoadedCloud() const {
    QMutexLocker lock(&m_mutex);
    return !m_loadedCloud.empty();
}

void CtReconstructionController::fillFromLoadedCloud(QObject* geometry) {
    auto* geom = qobject_cast<PointCloudGeometry*>(geometry);
    if (!geom) {
        qDebug() << "[CT] fillFromLoadedCloud: cannot cast geometry";
        return;
    }
    ct::PointCloud localCopy;
    {
        QMutexLocker lock(&m_mutex);
        if (m_loadedCloud.empty()) {
            qDebug() << "[CT] fillFromLoadedCloud: no loaded cloud (call loadPointCloud first)";
            return;
        }
        localCopy = m_loadedCloud;
    }
    geom->setPointCloud(localCopy);
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
#ifdef CT_CONSUMER_BUILD
        // Consumer гейтит готовностью загруженных синограмм, не hasVolume.
        const bool ready_to_start = !m_loadedSinograms.empty();
        qDebug() << "[CT] startReconstruction: state running=" << m_running
                 << "hasSinograms=" << ready_to_start << "ready=" << m_ready;
        if (m_running || !ready_to_start) {
            qDebug() << "[CT] startReconstruction: ignored (need to load sinograms)";
            return;
        }
#else
        qDebug() << "[CT] startReconstruction: state running=" << m_running << "hasVolume=" << m_hasVolume << "ready=" << m_ready;
        if (m_running || !m_hasVolume) {
            qDebug() << "[CT] startReconstruction: ignored";
            return;
        }
#endif
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

#ifdef CT_CONSUMER_BUILD
    // ============================================================
    // Consumer-вариант: FBP по уже загруженным синограммам.
    // Этап I (forward Radon) полностью пропускается.
    // UI заполняется только Reconstruction (Original/Sinogram/Difference
    // не имеют смысла — исходного тома нет, а синограммы сами на входе).
    // ============================================================
    int backendId, filterId;
    ct::Volume loadedSino;
    std::vector<float> angles;
    {
        QMutexLocker lock(&m_mutex);
        backendId  = m_backendType;
        filterId   = m_filterType;
        loadedSino = m_loadedSinograms;     // копия
        angles     = m_loadedAngles;
        results.genTimeSec = m_genTimeSec;
    }

    if (loadedSino.empty()) {
        qDebug() << "[CT] computeAll: no sinograms loaded";
        return results;
    }

    const int nz   = static_cast<int>(loadedSino.depth);
    const int na   = static_cast<int>(loadedSino.height);
    const int bins = static_cast<int>(loadedSino.width);

    results.hasVolume = true;   // используется UI как «есть с чем работать»
    results.maxZ      = nz - 1;
    results.currentZ  = nz / 2;
    results.reconstructionImages = std::make_shared<std::vector<QImage>>(static_cast<size_t>(nz));
    // Остальные слоты Consumer не заполняет (UI у него их и не отображает).
    results.originalImages       = std::make_shared<std::vector<QImage>>();
    results.sinogramImages       = std::make_shared<std::vector<QImage>>();
    results.differenceImages     = std::make_shared<std::vector<QImage>>();

    ct::ReconstructionParams params = defaultReconstructionParams();
    if      (filterId == 0) params.filter = ct::ReconstructionParams::FilterType::Ramp;
    else if (filterId == 1) params.filter = ct::ReconstructionParams::FilterType::SheppLogan;
    else if (filterId == 2) params.filter = ct::ReconstructionParams::FilterType::Hamming;
    else if (filterId == 3) params.filter = ct::ReconstructionParams::FilterType::Cosine;
    else if (filterId == 4) params.filter = ct::ReconstructionParams::FilterType::Hann;
    else if (filterId == 5) params.filter = ct::ReconstructionParams::FilterType::Bartlett;
    params.num_angles = static_cast<size_t>(na);  // из загруженных данных

    auto backendImpl = ct::BackendFactory::create(
        static_cast<ct::BackendFactory::BackendType>(backendId));
    if (!backendImpl || !backendImpl->isAvailable()) {
        backendImpl = ct::BackendFactory::createBestAvailable();
    }
    qDebug() << "[CT] computeAll[Consumer]: backend="
             << QString::fromStdString(backendImpl->name())
             << "nz=" << nz << "na=" << na << "bins=" << bins;

    // ВАЖНО: m_ready=true ставится только В applyResult (после генерации
    // QImage'ов). Если поставить его раньше — UI делает первый fetch ДО того,
    // как изображения заполнены, получает пустой QImage, и провайдер кэширует
    // его. Повторный readyChanged уже не триггерит refetch, потому что URL
    // не изменился (updateTicker=0, currentZ тот же).

    const auto t_recon_start = std::chrono::steady_clock::now();
    ct::Volume reconstructed_volume;
    try {
        backendImpl->reconstructVolumeFromSinograms(
            loadedSino, angles, reconstructed_volume, params, nullptr);
    } catch (const std::exception& e) {
        qDebug() << "[CT] EXCEPTION in reconstructVolumeFromSinograms:" << e.what();
        return results;
    }
    const auto t_recon_end = std::chrono::steady_clock::now();

    results.sinogramTimeSec = 0.0;   // Радона не было
    const double backend_recon_ms = backendImpl->lastReconstructionTimeMs();
    if (backend_recon_ms > 0.0) {
        results.reconTimeSec = backend_recon_ms / 1000.0;
    } else {
        results.reconTimeSec = std::chrono::duration<double>(t_recon_end - t_recon_start).count();
    }
    qDebug() << "[CT] computeAll[Consumer]: FBP done in" << results.reconTimeSec << "s";

    // Генерация QImage только для реконструкции.
    #pragma omp parallel for
    for (int z = 0; z < nz; ++z) {
        const size_t zi = static_cast<size_t>(z);
        const ct::Slice reconstruction = reconstructed_volume.getSlice(zi);
        (*results.reconstructionImages)[zi] = sliceToImage(reconstruction, false);
    }

    results.maxDifference = 0.0f;   // Consumer не вычисляет diff
    results.ready   = true;          // ← без этого applyResult оставит m_ready=false,
                                     //   UI source станет "", slider/Export disabled.
    results.success = true;
    return results;

#else
    // ============================================================
    // Producer-вариант: оригинальный путь (load NPY → Radon → FBP → metrics).
    // ============================================================
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
#endif // !CT_CONSUMER_BUILD
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

#ifdef CT_PRODUCER_BUILD
// ============================================================
//  exportAllSinograms — Producer-only
// ============================================================
// Открывает file dialog, перепрогоняет computeSinogram по всем слайсам
// текущего тома (загруженного с диска) и пишет NPY shape (nz, na, bins) +
// sidecar JSON с метаданными.
//
// Пересчёт через computeSinogram — выбран пользователем (см. план). Минимум
// изменений в backend'ах, использует существующий путь Радона. На быстром
// Hybrid'е для 1024³ это ~10-15с в worker-thread'е.
void CtReconstructionController::exportAllSinograms() {
    if (!m_ready && !m_hasVolume) {
        qDebug() << "[CT] exportAllSinograms: no volume / no reconstruction yet";
        return;
    }

    const QString defaultName = QCoreApplication::applicationDirPath()
                                + "/data/output/sinograms_"
                                + QString::number(m_volumeSize) + ".npy";
    const QString npyPath = QFileDialog::getSaveFileName(
        nullptr, "Export Sinograms (NPY + sidecar .meta.json)",
        defaultName, "NumPy (*.npy);;All files (*)");
    if (npyPath.isEmpty()) return;

    // Запускаем в worker-thread'е через QtConcurrent — UI не подвисает.
    setRunning(true);
    auto* watcher = new QFutureWatcher<bool>(this);
    connect(watcher, &QFutureWatcher<bool>::finished, this, [this, watcher, npyPath]() {
        const bool ok = watcher->result();
        watcher->deleteLater();
        setRunning(false);
        qDebug() << "[CT] exportAllSinograms:" << (ok ? "OK" : "FAILED") << "→" << npyPath;
    });

    QFuture<bool> fut = QtConcurrent::run([this, npyPath]() -> bool {
        int backendId, filterId, volSize;
        {
            QMutexLocker lock(&m_mutex);
            backendId = m_backendType;
            filterId  = m_filterType;
            volSize   = m_volumeSize;
        }

        // Загружаем исходный том с диска (тот же, что использует Generate/Recon).
        const QString inputNpy = QCoreApplication::applicationDirPath()
                                 + "/data/output/synthetic_brain_hu_cxx_"
                                 + QString::number(volSize) + ".npy";
        ct::Volume volume;
        if (!ct::FileIO::loadVolumeNPY(inputNpy.toStdString(), volume, false)) {
            qDebug() << "[CT] exportAllSinograms: failed to load source volume" << inputNpy;
            return false;
        }

        const size_t nz = volume.depth;
        const size_t bins = volume.width;

        ct::ReconstructionParams params = defaultReconstructionParams();
        if      (filterId == 0) params.filter = ct::ReconstructionParams::FilterType::Ramp;
        else if (filterId == 1) params.filter = ct::ReconstructionParams::FilterType::SheppLogan;
        else if (filterId == 2) params.filter = ct::ReconstructionParams::FilterType::Hamming;
        else if (filterId == 3) params.filter = ct::ReconstructionParams::FilterType::Cosine;
        else if (filterId == 4) params.filter = ct::ReconstructionParams::FilterType::Hann;
        else if (filterId == 5) params.filter = ct::ReconstructionParams::FilterType::Bartlett;
        const size_t na = params.num_angles;

        auto backend = ct::BackendFactory::create(
            static_cast<ct::BackendFactory::BackendType>(backendId));
        if (!backend || !backend->isAvailable()) {
            backend = ct::BackendFactory::createBestAvailable();
        }
        qDebug() << "[CT] exportAllSinograms: using backend"
                 << QString::fromStdString(backend->name())
                 << "nz=" << nz << "na=" << na << "bins=" << bins;

        // Аллоцируем sinogram-том shape (nz, na, bins).
        // Для 1024×180×1024 float32 = 754 МБ. На consumer-RAM это ОК
        // (на хосте, не на GPU).
        ct::Volume sinoVol;
        sinoVol.assign(bins, na, nz, 0.0f);   // width=bins, height=na, depth=nz

        // Глобальные min/max HU по всему тому — для записи в meta.
        float globalMin =  std::numeric_limits<float>::max();
        float globalMax = -std::numeric_limits<float>::max();

        const auto t0 = std::chrono::steady_clock::now();
        for (size_t z = 0; z < nz; ++z) {
            ct::Sinogram sino = backend->computeSinogram(volume.getSlice(z), na, bins);
            // Копируем data (size = na*bins float) в нужный слой sinoVol.
            float* dst = &sinoVol.data[(size_t)z * na * bins];
            std::memcpy(dst, sino.data.data.data(), na * bins * sizeof(float));
            if (sino.original_min_hu < globalMin) globalMin = sino.original_min_hu;
            if (sino.original_max_hu > globalMax) globalMax = sino.original_max_hu;
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double secs = std::chrono::duration<double>(t1 - t0).count();
        qDebug() << "[CT] exportAllSinograms: forward Radon for" << nz
                 << "slices in" << secs << "s";

        // ensureDirectory нерекурсивен — создаём оба уровня.
        const QString appDir = QCoreApplication::applicationDirPath();
        ct::utils::ensureDirectory(toWStringBackslashPath(appDir + "/data"));
        ct::utils::ensureDirectory(toWStringBackslashPath(appDir + "/data/output"));

        // 1. NPY (тот же layout, что у обычного volume — depth=nz, height=na, width=bins).
        if (!ct::FileIO::saveVolumeNPY(sinoVol, npyPath.toStdString())) {
            qDebug() << "[CT] exportAllSinograms: failed to write NPY" << npyPath;
            return false;
        }

        // 2. Sidecar JSON с метаданными.
        QJsonObject meta;
        meta["version"] = 1;
        meta["depth"]   = (int)nz;
        meta["num_angles"]   = (int)na;
        meta["detector_bins"] = (int)bins;
        meta["detector_spacing_mm"] = 1.0;
        meta["original_min_hu"] = globalMin;
        meta["original_max_hu"] = globalMax;
        QJsonArray angles;
        const float step = 180.0f / (float)na;
        for (size_t a = 0; a < na; ++a) angles.append(step * (double)a);
        meta["angles_deg"] = angles;

        QString jsonPath = npyPath;
        if (jsonPath.endsWith(".npy", Qt::CaseInsensitive)) {
            jsonPath.chop(4);
        }
        jsonPath += ".meta.json";
        QFile jf(jsonPath);
        if (!jf.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
            qDebug() << "[CT] exportAllSinograms: failed to write JSON sidecar" << jsonPath;
            return false;
        }
        jf.write(QJsonDocument(meta).toJson(QJsonDocument::Indented));
        jf.close();
        qDebug() << "[CT] exportAllSinograms: wrote NPY + meta.json";
        return true;
    });
    watcher->setFuture(fut);
}
#endif // CT_PRODUCER_BUILD

#ifdef CT_CONSUMER_BUILD
// ============================================================
//  loadSinograms — Consumer-only
// ============================================================
bool CtReconstructionController::hasSinograms() const {
    QMutexLocker lock(&m_mutex);
    return !m_loadedSinograms.empty();
}

bool CtReconstructionController::loadSinograms() {
    // Стартовая директория для QFileDialog. Порядок поиска:
    //   1. Последняя выбранная пользователем папка (QSettings, переживает перенос .exe)
    //   2. <exe>/data/output (типовой layout dev-сборки)
    //   3. Домашняя папка пользователя (гарантированно существует)
    QSettings settings(QStringLiteral("QtCTReconstruction"),
                       QStringLiteral("Consumer"));
    QString defaultDir = settings.value(QStringLiteral("lastSinogramDir")).toString();
    if (defaultDir.isEmpty() || !QDir(defaultDir).exists()) {
        const QString devDir = QCoreApplication::applicationDirPath() + "/data/output";
        defaultDir = QDir(devDir).exists()
                     ? devDir
                     : QDir::homePath();
    }

    const QString path = QFileDialog::getOpenFileName(
        nullptr, "Load Sinograms (NPY)", defaultDir,
        "NumPy sinograms (*.npy);;All files (*)");
    if (path.isEmpty()) return false;

    // Запоминаем выбранную папку для следующего запуска.
    settings.setValue(QStringLiteral("lastSinogramDir"),
                      QFileInfo(path).absolutePath());

    ct::Volume vol;
    if (!ct::FileIO::loadVolumeNPY(path.toStdString(), vol, /*input_is_xyz_layout=*/false)) {
        qDebug() << "[CT] loadSinograms: failed to parse" << path;
        return false;
    }
    const size_t nz   = vol.depth;
    const size_t na   = vol.height;
    const size_t bins = vol.width;
    if (nz == 0 || na == 0 || bins == 0) {
        qDebug() << "[CT] loadSinograms: empty volume in" << path;
        return false;
    }
    qDebug() << "[CT] loadSinograms: NPY shape" << nz << "×" << na << "×" << bins;

    // -------- Попытка загрузить sidecar JSON --------
    std::vector<float> angles;
    float spacing = 1.0f;
    float minHu = std::numeric_limits<float>::max();
    float maxHu = -std::numeric_limits<float>::max();
    bool gotMetaMinMax = false;

    QString jsonPath = path;
    if (jsonPath.endsWith(".npy", Qt::CaseInsensitive)) jsonPath.chop(4);
    jsonPath += ".meta.json";

    QFile jf(jsonPath);
    if (jf.open(QIODevice::ReadOnly)) {
        const QByteArray bytes = jf.readAll();
        QJsonParseError err;
        const QJsonDocument doc = QJsonDocument::fromJson(bytes, &err);
        if (err.error == QJsonParseError::NoError && doc.isObject()) {
            const QJsonObject obj = doc.object();
            if (obj.contains("detector_spacing_mm")) {
                spacing = (float)obj["detector_spacing_mm"].toDouble(1.0);
            }
            if (obj.contains("original_min_hu") && obj.contains("original_max_hu")) {
                minHu = (float)obj["original_min_hu"].toDouble(0.0);
                maxHu = (float)obj["original_max_hu"].toDouble(0.0);
                gotMetaMinMax = true;
            }
            if (obj.contains("angles_deg") && obj["angles_deg"].isArray()) {
                const QJsonArray arr = obj["angles_deg"].toArray();
                angles.reserve(arr.size());
                for (const auto& v : arr) angles.push_back((float)v.toDouble(0.0));
            }
            qDebug() << "[CT] loadSinograms: meta.json loaded (spacing=" << spacing
                     << "min_hu=" << (gotMetaMinMax ? minHu : 0.0f)
                     << "max_hu=" << (gotMetaMinMax ? maxHu : 0.0f)
                     << "angles count=" << (int)angles.size() << ")";
        } else {
            qDebug() << "[CT] loadSinograms: meta.json parse error:" << err.errorString();
        }
    } else {
        qDebug() << "[CT] loadSinograms: no meta.json sidecar (defaults will be used)";
    }

    // Если углов нет / неверный размер — равномерно [0, 180).
    if (angles.size() != na) {
        if (!angles.empty()) {
            qDebug() << "[CT] loadSinograms: meta angles count" << (int)angles.size()
                     << "!= na" << (int)na << "; falling back to uniform";
        }
        angles.assign(na, 0.0f);
        const float step = 180.0f / (float)na;
        for (size_t a = 0; a < na; ++a) angles[a] = step * (float)a;
    }

    // Если min/max не пришли — посчитаем из данных (медленно, но точно).
    if (!gotMetaMinMax) {
        for (float v : vol.data) {
            if (v < minHu) minHu = v;
            if (v > maxHu) maxHu = v;
        }
    }

    {
        QMutexLocker lock(&m_mutex);
        m_loadedSinograms = std::move(vol);
        m_loadedAngles    = std::move(angles);
        m_loadedSpacingMm = spacing;
        m_loadedMinHu     = minHu;
        m_loadedMaxHu     = maxHu;
        m_volumeSize      = (int)bins;   // output size реконструкции = bins
        // Старая реконструкция (если была) больше не валидна.
        m_ready = false;
        if (m_originalImagesPtr) m_originalImagesPtr->clear();
        if (m_sinogramImagesPtr) m_sinogramImagesPtr->clear();
        if (m_reconstructionImagesPtr) m_reconstructionImagesPtr->clear();
        if (m_differenceImagesPtr) m_differenceImagesPtr->clear();
        // m_hasVolume в Consumer не используется — гейтит hasSinograms.
    }

    emit hasSinogramsChanged();
    emit volumeSizeChanged();
    emit readyChanged();
    qDebug() << "[CT] loadSinograms: ready (nz=" << (int)nz << ", na=" << (int)na
             << ", bins=" << (int)bins << ")";
    return true;
}
#endif // CT_CONSUMER_BUILD
