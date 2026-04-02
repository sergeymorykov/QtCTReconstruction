#pragma once

// Порядок включения важен: Generator3D.h (Params) → IVolumeGenerator.h (интерфейс)
#include "Generator3D.h"
#include "IVolumeGenerator.h"

namespace ct {

class Generator3DGPU : public IVolumeGenerator {
public:
    std::string name() const override { return "GPU (CUDA)"; }
    bool isAvailable() const override;

    Volume generateBrainHU(const Generator3D::Params& params) override;
};

} // namespace ct
