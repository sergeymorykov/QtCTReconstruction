#pragma once

// HybridBackend — ультимативный гибрид OpenMP (CPU) + CUDA (GPU).
//
// Архитектура пайплайна (формулы (2), (5)-(8); air-skip из статьи
// "fast hybrid CPU/GPU CT reconstruction with air skipping"):
//
//   Этап I  (GPU)  : Radon-проекция всего объёма → синограмма
//   Этап II (CHUNK-loop, CPU∥GPU):
//       per chunk i:
//         (a) GPU stream_filt: pack + FFT + filter + iFFT + scale  -> pad[i%2]
//         (b) GPU stream_copy: D2H pad[i%2] → h_filt[i%2]
//         (c) CPU std::async : computeAirBoundariesCPU по ФИЛЬТРОВАННОЙ синограмме
//                              (запускается, как только готово (b))
//         (d) GPU stream_bp  : backproj чанка (i-1) с air-skip, используя
//                              его собственные pad[(i-1)%2] и d_bnd[(i-1)%2]
//                              ← бежит ПАРАЛЛЕЛЬНО с (a)(b)(c) текущей итерации
//         (e) GPU stream_bp  : D2H output чанка (i-1)
//
//   Реальный overlap CPU↔GPU:
//     CPU computeAirBoundariesCPU(i)   ║   GPU backproj(i-1) + filter(i+1)
//
// Для single-slice вызовов (computeSinogram / reconstructSlice) делегирует
// OpenMP-CPU-путям: PCIe-передача одного среза нерентабельна.

#include "IReconstructionBackend.h"

#include <cufft.h>
#include <cuda_runtime.h>
#include <mutex>

namespace ct {

class HybridBackend : public IReconstructionBackend {
public:
    HybridBackend() = default;
    ~HybridBackend();

    std::string name() const override { return "Hybrid (CUDA + OpenMP)"; }
    bool isAvailable() const override;

    Sinogram computeSinogram(const Buffer2D& slice, size_t num_angles,
                             size_t detector_bins, bool use_parallel = true) override;
    Buffer2D reconstructSlice(const Sinogram& sinogram, size_t output_size,
                              const ReconstructionParams& params) override;

    void reconstructVolume(const Volume& input_volume,
                           Volume& out_reconstruction,
                           const ReconstructionParams& params,
                           std::function<void(int, const Buffer2D&)> onSliceDone) override;

    PointCloud extractPointCloud(const Volume& vol, float threshold) override;
    double lastSinogramTimeMs()      const override { return m_lastSinogramTimeMs; }
    double lastReconstructionTimeMs() const override { return m_lastReconstructionTimeMs; }

private:
    mutable std::recursive_mutex m_mutex;
    mutable double m_lastSinogramTimeMs      = 0.0;
    mutable double m_lastReconstructionTimeMs = 0.0;

    // ---- Device workspace (persistent, lazy) ----
    mutable float*        m_d_vol_in     = nullptr;   // полный том
    mutable float*        m_d_vol_out    = nullptr;   // полный том (recon)
    mutable float*        m_d_all_sinos  = nullptr;   // [na × nz × bins]
    // Два pad-буфера для ping-pong между filter и backproj
    mutable float*        m_d_projs_pad[2]    = { nullptr, nullptr };
    mutable cufftComplex* m_d_spectrum   = nullptr;   // используется по очереди стримом stream_filt
    mutable float*        m_d_filter     = nullptr;
    mutable float*        m_d_cos        = nullptr;
    mutable float*        m_d_sin        = nullptr;
    mutable float*        m_d_min_hu     = nullptr;   // [nz] — заполняется на GPU в Этапе I, читается в Этапе II
    mutable float*        m_d_span       = nullptr;   // [nz] — то же
    mutable float*        m_d_slice_minmax = nullptr; // [2*nz] — raw (vmin, vmax) per slice, intermediate
    // Boundary tables на устройстве, ping-pong по slot'у
    mutable int*          m_d_umin[2]    = { nullptr, nullptr };
    mutable int*          m_d_umax[2]    = { nullptr, nullptr };

    // ---- Pinned host buffers (WC) для D2H/H2D без kernel'ного stall ----
    mutable float*        m_h_filt[2]     = { nullptr, nullptr };  // D2H filtered chunk
    mutable int*          m_h_umin[2]     = { nullptr, nullptr };
    mutable int*          m_h_umax[2]     = { nullptr, nullptr };
    mutable size_t        m_filtCap       = 0;       // ёмкость одного h_filt в float'ах
    mutable size_t        m_bndCap        = 0;       // ёмкость одного h_umin/h_umax в int'ах

    // ---- Backproj 3D texture cache (memory pool) [P3] ----
    // Layout: extent (sq, chunk_nz, na). Адресация tex3D(u, v_row, a).
    // Соседние углы — соседние z-слои текстуры → лучший cache hit-rate, чем
    // 2D-layout (sq, na*chunk_nz), где переход a→a+1 прыгал на chunk_nz строк.
    mutable cudaArray_t         m_bpArray3D = nullptr;
    mutable cudaTextureObject_t m_bpTexObj  = 0;
    mutable size_t              m_bpArrayW  = 0;   // sq
    mutable size_t              m_bpArrayH  = 0;   // chunk_nz
    mutable size_t              m_bpArrayD  = 0;   // na

    // ---- Capacities ----
    mutable size_t m_volCap         = 0;
    mutable size_t m_sinoCap        = 0;
    mutable size_t m_projPadCap     = 0;   // ёмкость одного из pad-буферов
    mutable size_t m_spectrumCap    = 0;
    mutable size_t m_filterSize     = 0;
    mutable size_t m_trigAngles     = 0;
    mutable size_t m_depthCap       = 0;
    mutable size_t m_currentProjPad = 0;
    mutable ReconstructionParams::FilterType m_filterType = ReconstructionParams::FilterType::SheppLogan;

    // cuFFT
    mutable cufftHandle m_planR2C  = 0;
    mutable cufftHandle m_planC2R  = 0;
    mutable size_t      m_planBatch  = 0;
    mutable size_t      m_planPadded = 0;

    // ---- Streams + events для async ping-pong ----
    mutable cudaStream_t m_stream_filt = nullptr;   // filter+iFFT
    mutable cudaStream_t m_stream_copy = nullptr;   // D2H filtered, H2D bnd
    mutable cudaStream_t m_stream_bp   = nullptr;   // backproject
    // event_filt_done[slot] — filter chunk завершил запись pad[slot]
    // event_d2h_done [slot] — D2H filtered chunk в h_filt[slot] завершён (CPU может читать)
    // event_bp_done  [slot] — backproj прочитал pad[slot]; filter может его перезаписать
    mutable cudaEvent_t  m_event_filt_done[2] = { nullptr, nullptr };
    mutable cudaEvent_t  m_event_d2h_done [2] = { nullptr, nullptr };
    mutable cudaEvent_t  m_event_bp_done  [2] = { nullptr, nullptr };
    mutable bool         m_streamsInited      = false;

    // Лёгкий кэш для углов
    mutable float m_cachedFirstAngle = 0.0f;
    mutable float m_cachedLastAngle  = 0.0f;

    // Single-slice scratch (для computeSinogram при UI-постобработке —
    // controller вызывает его ~512 раз, CPU-Радон бы занял ~15 сек).
    mutable float* m_d_single_slice  = nullptr;
    mutable float* m_d_single_sino   = nullptr;
    mutable size_t m_singleSliceCap  = 0;   // pixels
    mutable size_t m_singleSinoCap   = 0;   // floats

    // ---- Helpers ----
    void ensureWorkspace(size_t w, size_t h, size_t d,
                         size_t num_angles, size_t detector_bins,
                         size_t chunk_z) const;
    void ensureFilter(size_t padded_size, ReconstructionParams::FilterType type) const;
    void ensurePlans(size_t batch, size_t padded_size) const;
    void ensureTrigTables(const std::vector<float>& angles_deg) const;
    void ensureBackprojTexture3D(size_t sq, size_t chunk_nz, size_t na) const;
    void ensureStreams() const;
    void releaseAll() const;
};

} // namespace ct
