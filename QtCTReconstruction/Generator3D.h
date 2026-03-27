#pragma once

#include "CTTypes.h"

#include <array>
#include <cstdint>

namespace ct {

class Generator3D {
public:
    struct Params {
        std::array<size_t, 3> shape = {128, 128, 64}; // [x, y, z]
        float brain_radius_mm = 50.0f;
        size_t num_ellipsoids = 120;
        uint32_t seed = 42;
    };

    static Volume generateBrainHU(const Params& params);
};

} // namespace ct
