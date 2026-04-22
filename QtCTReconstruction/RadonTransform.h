#pragma once

#include "CTTypes.h"

namespace ct {

class RadonTransform {
public:
    static Sinogram forward(const Slice& slice, size_t num_angles, size_t detector_bins, float min_hu = 0.0f, float max_hu = 1.0f);
};

} // namespace ct
