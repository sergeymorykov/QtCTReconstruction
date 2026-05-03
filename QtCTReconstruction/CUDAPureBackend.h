#pragma once

#include "IReconstructionBackend.h"
#include <cufft.h>
#include <mutex>

namespace ct {

// Pure-GPU backend: uploads the full volume once, runs all pipeline stages
// (sinogram, filter, backprojection) on GPU, downloads the result once.
// Contrast with CUDABackend which transfers one slice at a time.
class CUDAPureBackend : public IReconstructionBackend {
public:
    CUDAPureBackend() = default;
    ~CUDAPureBackend();

    std::string name() const override { return "CUDA Pure (Full-GPU)"; }
    bool isAvailable() const override;

    Sinogram computeSinogram(const Buffer2D& slice, size_t num_angles,
                             size_t detector_bins, bool use_parallel = true) override;
    Buffer2D reconstructSlice(const Sinogram& sinogram, size_t output_size,
                              const ReconstructionParams& params) override;

    void reconstructVolume(const Volume& input_volume,
                           Volume& out_reconstruction,
                           const ReconstructionParams& params,
                           std::function<void(int slice_idx, const Buffer2D&)> onSliceDone) override;

    PointCloud extractPointCloud(const Volume& vol, float threshold) override;
    double lastSinogramTimeMs() const override { return m_lastSinogramTimeMs; }
    double lastReconstructionTimeMs() const override { return m_lastReconstructionTimeMs; }

private:
    mutable std::recursive_mutex m_mutex;
    mutable double m_lastSinogramTimeMs = 0.0;
    mutable double m_lastReconstructionTimeMs = 0.0;

    // Persistent GPU workspace (lazy-allocated, resized on demand)
    mutable float*        m_d_vol_in    = nullptr;  // input volume
    mutable float*        m_d_vol_out   = nullptr;  // output volume
    mutable float*        m_d_all_sinos = nullptr;  // [depth × num_angles × detector_bins]
    mutable float*        m_d_projs_pad = nullptr;  // [num_angles × depth × proj_size_padded]
    mutable cufftComplex* m_d_spectrum  = nullptr;  // [num_angles × depth × complex_size]
    mutable float*        m_d_filter    = nullptr;
    mutable float*        m_d_cos       = nullptr;
    mutable float*        m_d_sin       = nullptr;
    mutable float*        m_d_min_hu    = nullptr;  // per-slice [depth]
    mutable float*        m_d_span      = nullptr;  // per-slice [depth]
    // Air-skipping boundary tables: u_min/u_max per (angle, z).
    // Layout: row-major [angle * depth + z].
    mutable int*          m_d_umin      = nullptr;
    mutable int*          m_d_umax      = nullptr;
    mutable size_t        m_boundCap    = 0;

    mutable size_t m_volCap      = 0;
    mutable size_t m_sinoCap     = 0;
    mutable size_t m_projPadCap  = 0;
    mutable size_t m_spectrumCap = 0;
    mutable size_t m_filterSize  = 0;
    mutable size_t m_trigAngles  = 0;
    mutable size_t m_depthCap    = 0;
    mutable ReconstructionParams::FilterType m_filterType = ReconstructionParams::FilterType::SheppLogan;

    mutable cufftHandle m_planR2C  = 0;
    mutable cufftHandle m_planC2R  = 0;
    mutable size_t      m_planBatch   = 0;
    mutable size_t      m_planPadded  = 0;

    // Лёгкий кэш-сигнатура для ensureTrigTables (не храним полный вектор —
    // его копирование в Debug-билде на больших объёмах вызывало access violation
    // из-за фрагментации хипа).
    mutable float m_cachedFirstAngle = 0.0f;
    mutable float m_cachedLastAngle  = 0.0f;

    void ensureWorkspace(size_t w, size_t h, size_t d, size_t num_angles, size_t detector_bins) const;
    void ensureFilter(size_t padded_size, ReconstructionParams::FilterType type) const;
    void ensurePlans(size_t batch, size_t padded_size) const;
    void ensureTrigTables(const std::vector<float>& angles_deg) const;
    void releaseAll() const;
};

} // namespace ct
