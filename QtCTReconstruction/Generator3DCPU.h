#pragma once

// Порядок включения важен: Generator3D.h (Params) → IVolumeGenerator.h (интерфейс)
#include "Generator3D.h"
#include "IVolumeGenerator.h"

namespace ct {

class Generator3DCPU : public IVolumeGenerator {
public:
    std::string name() const override { return "CPU (OpenMP)"; }
    bool isAvailable() const override { return true; }

    Volume generateBrainHU(const Generator3D::Params& params) override;
};

} // namespace ct
