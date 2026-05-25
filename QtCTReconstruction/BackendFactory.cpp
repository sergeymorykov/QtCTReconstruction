#include "IReconstructionBackend.h"
#ifdef CT_HAS_SERIAL_BACKEND
#include "SerialBackend.h"
#endif
#include "OpenMPBackend.h"
#include "CUDABackend.h"
#include "HybridBackend.h"

namespace ct {

// Дефолтная реализация: per-slice loop через reconstructSlice. Подходит для
// любого backend'а, не требует доступа к внутреннему конвейеру. Бэкенды могут
// переопределить для batched-оптимизации (Hybrid делает это, пропуская
// Этап I; CUDA — pinned-staging как в reconstructVolume).
void IReconstructionBackend::reconstructVolumeFromSinograms(
    const Volume& sinograms,
    const std::vector<float>& angles_deg,
    Volume& out_reconstruction,
    const ReconstructionParams& params,
    std::function<void(int, const Buffer2D&)> onSliceDone)
{
    // sinograms layout: depth=nz, height=na, width=bins (как у reconstructVolume
    // выходной том — Z строк = слайсы синограммы).
    const size_t nz   = sinograms.depth;
    const size_t na   = sinograms.height;
    const size_t bins = sinograms.width;
    if (nz == 0 || na == 0 || bins == 0) return;

    // Размер тома реконструкции = bins (квадратный output по детектору).
    const size_t output_size = bins;
    out_reconstruction.assign(output_size, output_size, nz, 0.0f);

    // Чтобы избежать копирования всего тома в локальный Sinogram, делаем
    // лёгкий wrapper: на каждый z берём строку (na × bins) через getSlice().
    // getSlice копирует данные (Buffer2D), но это всего ~na*bins float — для
    // 180×1024 ≈ 720 КБ, приемлемо.
    for (size_t z = 0; z < nz; ++z) {
        Sinogram s;
        s.data = sinograms.getSlice(z);       // shape (bins, na)
        s.angles_deg = angles_deg;
        s.detector_spacing_mm = 1.0f;
        // min/max HU оценим по самой синограмме — backend будет использовать
        // их для денормализации (если он этот путь использует).
        float vmin = s.data.data.empty() ? 0.0f : s.data.data[0];
        float vmax = vmin;
        for (float v : s.data.data) {
            if (v < vmin) vmin = v;
            if (v > vmax) vmax = v;
        }
        s.original_min_hu = vmin;
        s.original_max_hu = vmax;

        Buffer2D recon = reconstructSlice(s, output_size, params);
        out_reconstruction.setSlice(z, recon);

        if (onSliceDone) onSliceDone(static_cast<int>(z), recon);
    }
}

std::shared_ptr<IReconstructionBackend> BackendFactory::create(BackendType type) {
#ifdef CT_HAS_SERIAL_BACKEND
    if (type == BackendType::Serial) {
        return std::make_shared<SerialBackend>();
    } else
#endif
    if (type == BackendType::OpenMP) {
        return std::make_shared<OpenMPBackend>();
    } else if (type == BackendType::CUDA) {
        return std::make_shared<CUDABackend>();
    } else if (type == BackendType::Hybrid) {
        return std::make_shared<HybridBackend>();
    }
    return nullptr;
}

std::shared_ptr<IReconstructionBackend> BackendFactory::createBestAvailable() {
    auto cudaBackend = create(BackendType::CUDA);
    if (cudaBackend && cudaBackend->isAvailable()) {
        return cudaBackend;
    }
    return create(BackendType::OpenMP);
}

} // namespace ct
