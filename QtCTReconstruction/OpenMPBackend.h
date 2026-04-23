#pragma once

#include "IReconstructionBackend.h"

namespace ct {

class OpenMPBackend : public IReconstructionBackend {
public:
    std::string name() const override { return "OpenMP (CPU)"; }
    bool isAvailable() const override { return true; }

    Sinogram computeSinogram(const Buffer2D& slice, size_t num_angles, size_t detector_bins) override;
    Buffer2D reconstructSlice(const Sinogram& sinogram, size_t output_size, const ReconstructionParams& params) override;

    void reconstructVolume(const Volume& input_volume, 
                           Volume& out_reconstruction,
                           const ReconstructionParams& params,
                           std::function<void(int slice_idx, const Buffer2D& recon_slice)> onSliceDone) override;

    PointCloud extractPointCloud(const Volume& vol, float threshold) override;
    double lastSinogramTimeMs() const override { return m_lastSinogramTimeMs; }

private:
    Sinogram computeSinogram(const float* slice_data, size_t w, size_t h, size_t num_angles, size_t detector_bins);
    double m_lastSinogramTimeMs = 0.0;
};

} // namespace ct
