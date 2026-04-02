#pragma once

#include "IReconstructionBackend.h"

namespace ct {

class CUDABackend : public IReconstructionBackend {
public:
    std::string name() const override { return "CUDA (GPU)"; }
    bool isAvailable() const override;

    Sinogram computeSinogram(const Buffer2D& slice, size_t num_angles, size_t detector_bins) override;
    Buffer2D reconstructSlice(const Sinogram& sinogram, size_t output_size, const ReconstructionParams& params) override;

    void reconstructVolume(const Volume& input_volume, 
                           Volume& out_reconstruction,
                           const ReconstructionParams& params,
                           std::function<void(int slice_idx, const Buffer2D& recon_slice)> onSliceDone) override;

    PointCloud extractPointCloud(const Volume& vol, float threshold) override;
};

} // namespace ct
