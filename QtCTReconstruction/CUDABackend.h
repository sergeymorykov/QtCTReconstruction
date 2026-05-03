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

    // ---- [P1] Pinned host memory для двойной буферизации ----
    // Два буфера: пока stream A копирует slice[z+1], stream B реконструирует slice[z].
    // cudaHostAllocPortable | cudaHostAllocWriteCombined: пропускная способность H→D
    // вырастает с ~6 до ~12 ГБ/с на PCIe 3.0 x16 по сравнению с pageable.
    mutable float*  m_h_slice_A = nullptr;   // pinned host буфер А
    mutable float*  m_h_slice_B = nullptr;   // pinned host буфер B
    mutable size_t  m_pinnedSliceCap = 0;    // размер (пикселей) каждого буфера

    // ---- [P1] CUDA streams для двойной буферизации ----
    // stream A: копирует данные; stream B: выполняет ядра.
    // Роли чередуются (ping-pong) между итерациями.
    mutable cudaStream_t m_stream_A = nullptr;
    mutable cudaStream_t m_stream_B = nullptr;
    mutable cudaEvent_t  m_event_A  = nullptr;  // сигнализирует о готовности slice A
    mutable cudaEvent_t  m_event_B  = nullptr;  // сигнализирует о готовности slice B
    // События "stream_compute прочитал m_d_vol_in" — нужны чтобы stream_copy
    // не перезаписал буфер до завершения ядер (без них был off-by-one bug:
    // recon[z] получался по данным slice[z+1]).
    mutable cudaEvent_t  m_event_compute_A = nullptr;
    mutable cudaEvent_t  m_event_compute_B = nullptr;

    // ---- Filter cache ----
    mutable float*  m_d_filter      = nullptr;
    mutable size_t  m_filterSize    = 0;
    mutable ReconstructionParams::FilterType
                    m_filterType    = ReconstructionParams::FilterType::SheppLogan;

    // ---- FFT workspace ----
    mutable cufftComplex* m_d_spectrum = nullptr;
    mutable size_t        m_spectrumSize = 0;

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
    // [P1] Гарантирует наличие pinned-буферов и CUDA streams
    void ensurePinnedAndStreams(size_t slice_pixels) const;
    void releaseCache() const;
};

} // namespace ct
