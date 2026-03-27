#pragma once

#include "CTTypes.h"

namespace ct {

class FilteredBackprojection {
public:
    static Slice reconstruct(const Sinogram& sinogram, size_t output_size, const ReconstructionParams& params);
};

} // namespace ct
