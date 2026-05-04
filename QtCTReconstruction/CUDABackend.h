#pragma once

#include "IReconstructionBackend.h"
#include <cufft.h>
#include <cuda_runtime.h>
#include <mutex>

namespace ct {

class CUDABackend : public IReconstructionBackend {
public:
    CUDABackend() = default;
    ~CUDABackend();

    std::string name() const override { return "CUDA (GPU)"; }
    bool isAvailable() const override;

    Sinogram computeSinogram(const Buffer2D& slice, size_t num_angles, size_t detector_bins, bool use_parallel = true) override;
    Buffer2D reconstructSlice(const Sinogram& sinogram, size_t output_size, const ReconstructionParams& params) override;

    void reconstructVolume(const Volume& input_volume, 
                           Volume& out_reconstruction,
                           const ReconstructionParams& params,
                           std::function<void(int slice_idx, const Buffer2D& recon_slice)> onSliceDone) override;

    PointCloud extractPointCloud(const Volume& vol, float threshold) override;
    double lastSinogramTimeMs() const override { return m_lastSinogramTimeMs; }
    
    // Пакетная очистка кэша
    void clearWorkspace() const;

private:
    mutable std::recursive_mutex m_mutex;
    mutable double m_lastSinogramTimeMs = 0.0;

    // ---- Device workspace ----
    mutable float* m_d_vol_in  = nullptr;
    mutable float* m_d_vol_out = nullptr;
    mutable float* m_d_sino    = nullptr;
    mutable size_t m_volSize   = 0;
    mutable size_t m_sinoSize  = 0;

    // ---- Pinned host memory ----
    // cudaHostAllocPortable | cudaHostAllocWriteCombined: пропускная способность H→D
    // ~12 ГБ/с на PCIe 3.0 x16 против ~6 ГБ/с pageable memory.
    mutable float*  m_h_slice_A = nullptr;   // pinned host буфер для H→D одного среза
    mutable size_t  m_pinnedSliceCap = 0;    // размер (пикселей)

    // ---- Filter cache ----
    mutable float*  m_d_filter      = nullptr;
    mutable size_t  m_filterSize    = 0;
    mutable ReconstructionParams::FilterType
                    m_filterType    = ReconstructionParams::FilterType::SheppLogan;

    // ---- FFT workspace ----
    mutable cufftComplex* m_d_spectrum = nullptr;
    mutable size_t        m_spectrumSize = 0;

    // ---- Backprojection texture cache (пункт 3.1 из плана оптимизаций) ----
    // На 256³ старая схема создавала/удаляла cudaArray + textureObject 256 раз
    // (~10-20 мс/срез × 256 = ~3-5 сек overhead). Теперь аллоцируется один раз
    // под максимальный размер и переиспользуется через cudaMemcpy2DToArray.
    mutable cudaArray_t         m_textureArray   = nullptr;
    mutable cudaTextureObject_t m_texObj         = 0;
    mutable size_t              m_texArrayWidth  = 0;  // = num_angles
    mutable size_t              m_texArrayHeight = 0;  // = square_bins

    // ---- cuFFT plans ----
    mutable cufftHandle m_planR2C = 0;
    mutable cufftHandle m_planC2R = 0;
    mutable size_t  m_planAngles  = 0;
    mutable size_t  m_planPadded  = 0;

    // ---- Trig tables ----
    mutable float*  m_d_cos       = nullptr;
    mutable float*  m_d_sin       = nullptr;
    mutable size_t  m_trigAngles  = 0;
    // Лёгкая кэш-сигнатура: храним count + first + last вместо полного вектора.
    // Полный std::vector::operator= в Debug-билде падал с access violation
    // на больших объёмах из-за heap-фрагментации.
    mutable float m_cachedFirstAngle = 0.0f;
    mutable float m_cachedLastAngle  = 0.0f;

    // ---- Helpers ----
    void ensureFilter(size_t padded_size, ReconstructionParams::FilterType type) const;
    void ensurePlans(size_t num_angles, size_t padded_size) const;
    void ensureTrigTables(const std::vector<float>& angles_deg) const;
    void ensureWorkspace(size_t w, size_t h, size_t d, size_t num_angles, size_t bins) const;
    // Гарантирует наличие pinned-буфера для H→D трансфера одного среза
    void ensurePinnedAndStreams(size_t slice_pixels) const;
    // Гарантирует наличие cudaArray и textureObject под backprojection.
    // Пересоздаёт их только при изменении размеров (пункт 3.1 из плана).
    void ensureTextureCache(size_t num_angles, size_t square_bins) const;
    void releaseCache() const;
};

} // namespace ct
