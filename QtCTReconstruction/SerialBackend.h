#pragma once

#include "IReconstructionBackend.h"

namespace ct {

// Эталонная однопоточная CPU-реализация без OpenMP, без CUDA и без SIMD-intrinsics.
// Назначение:
//   1. Базис для коэффициента ускорения (знаменатель A_OMP, A_GPU).
//   2. Регрессионные тесты корректности (детерминированный порядок суммирования).
//   3. Портативность: собирается без каких-либо внешних зависимостей.
//   4. Учебная ценность: прямое соответствие формулам раздела 2 статьи.
//
// Управляется флагом CMake -DBUILD_SERIAL_BACKEND=ON (по умолчанию ON в Debug).
class SerialBackend : public IReconstructionBackend {
public:
    std::string name() const override { return "Serial (CPU, 1 thread)"; }
    bool isAvailable() const override { return true; }

    // Формула (2): преобразование Радона, p(theta, s) = integral f(x,y) delta(x cos theta + y sin theta - s) dxdy
    Sinogram computeSinogram(const Buffer2D& slice, size_t num_angles, size_t detector_bins, bool use_parallel = true) override;

    // Формулы (5)-(8): фильтрация рамп-фильтром и обратная проекция
    Buffer2D reconstructSlice(const Sinogram& sinogram, size_t output_size, const ReconstructionParams& params) override;

    // Пакетная реконструкция тома: срез за срезом в одном потоке
    void reconstructVolume(const Volume& input_volume,
                           Volume& out_reconstruction,
                           const ReconstructionParams& params,
                           std::function<void(int slice_idx, const Buffer2D& recon_slice)> onSliceDone) override;

    PointCloud extractPointCloud(const Volume& vol, float threshold) override;
    double lastSinogramTimeMs() const override { return m_lastSinogramTimeMs; }

private:
    mutable double m_lastSinogramTimeMs = 0.0;
};

} // namespace ct
