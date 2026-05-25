#pragma once

#include "CTTypes.h"
#include <functional>
#include <memory>
#include <string>

namespace ct {

class IReconstructionBackend {
public:
    virtual ~IReconstructionBackend() = default;

    // Имя бэкенда для UI
    virtual std::string name() const = 0;

    // Доступен ли данный бэкенд на текущей системе
    virtual bool isAvailable() const = 0;

    // Выполнение преобразования Радона (создание синограммы из среза)
    virtual Sinogram computeSinogram(const Buffer2D& slice, size_t num_angles, size_t detector_bins, bool use_parallel = true) = 0;

    // Выполнение фильтрации и обратной проекции (реконструкция среза из синограммы)
    virtual Buffer2D reconstructSlice(const Sinogram& sinogram, size_t output_size, const ReconstructionParams& params) = 0;

    // Пакетная параллельная обработка тома с коллбеком для UI ("on-the-fly" update)
    // Функция запускает процесс и может быть асинхронной или блокирующей.
    virtual void reconstructVolume(const Volume& input_volume,
                                   Volume& out_reconstruction,
                                   const ReconstructionParams& params,
                                   std::function<void(int slice_idx, const Buffer2D& recon_slice)> onSliceDone) = 0;

    // Реконструкция тома из УЖЕ ГОТОВЫХ синограмм (используется Consumer-сборкой).
    // Этап I (forward Radon) пропускается; на вход — sinograms-том shape
    // (nz, na, bins) и список углов в градусах.
    //
    // Дефолтная реализация — per-slice loop через reconstructSlice. Backend'ы
    // могут переопределить для batched-оптимизации (CUDA/Hybrid делают это).
    virtual void reconstructVolumeFromSinograms(
        const Volume& sinograms,
        const std::vector<float>& angles_deg,
        Volume& out_reconstruction,
        const ReconstructionParams& params,
        std::function<void(int slice_idx, const Buffer2D& recon_slice)> onSliceDone);

    // Извлечение облака точек по порогу HU
    virtual PointCloud extractPointCloud(const Volume& vol, float threshold) = 0;

    // Время последнего этапа проецирования (мс)
    virtual double lastSinogramTimeMs() const = 0;

    // Время последнего этапа FBP-реконструкции (мс).
    // Default 0: переопределяется в бэкендах, где этапы измеряются раздельно.
    virtual double lastReconstructionTimeMs() const { return 0.0; }
};

// Простая фабрика для получения бэкендов
class BackendFactory {
public:
    enum class BackendType { Serial, OpenMP, CUDA, Hybrid };

    static std::shared_ptr<IReconstructionBackend> create(BackendType type);
    static std::shared_ptr<IReconstructionBackend> createBestAvailable();
};

} // namespace ct
