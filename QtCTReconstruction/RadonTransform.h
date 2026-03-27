#pragma once

#include "CTTypes.h"

namespace ct {

class RadonTransform {
public:
    static Sinogram forward(const Slice& slice, size_t num_angles, size_t detector_bins);
};

} // namespace ct
