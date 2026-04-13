#pragma once

#include "IReconstructionBackend.h"
#include <cufft.h>

namespace ct {

class CUDABackend : public IReconstructionBackend {
public:
    CUDABackend() = default;
    ~CUDABackend();

    std::string name() const override { return "CUDA (GPU)"; }
    bool isAvailable() const override;

    Sinogram computeSinogram(const Buffer2D& slice, size_t num_angles, size_t detector_bins) override;
    Buffer2D reconstructSlice(const Sinogram& sinogram, size_t output_size, const ReconstructionParams& params) override;

    void reconstructVolume(const Volume& input_volume, 
                           Volume& out_reconstruction,
                           const ReconstructionParams& params,
                           std::function<void(int slice_idx, const Buffer2D& recon_slice)> onSliceDone) override;

    PointCloud extractPointCloud(const Volume& vol, float threshold) override;

private:
    // ---- Кэш: фильтр ----
    mutable float*  m_d_filter      = nullptr;
    mutable size_t  m_filterSize    = 0;
    mutable ReconstructionParams::FilterType
                    m_filterType    = ReconstructionParams::FilterType::SheppLogan;

    // ---- Кэш: cuFFT планы ----
    mutable cufftHandle m_planR2C = 0;
    mutable cufftHandle m_planC2R = 0;
    mutable size_t  m_planAngles  = 0;
    mutable size_t  m_planPadded  = 0;

    // ---- Кэш: тригонометрические таблицы ----
    mutable float*  m_d_cos       = nullptr;
    mutable float*  m_d_sin       = nullptr;
    mutable size_t  m_trigAngles  = 0;
    mutable std::vector<float> m_cachedAnglesDeg;

    // ---- Вспомогательные методы ----
    void ensureFilter(size_t padded_size, ReconstructionParams::FilterType type) const;
    void ensurePlans(size_t num_angles, size_t padded_size) const;
    void ensureTrigTables(const std::vector<float>& angles_deg) const;
    void releaseCache() const;
};

} // namespace ct
