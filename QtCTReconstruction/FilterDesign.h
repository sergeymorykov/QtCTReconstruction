#pragma once

#include "CTTypes.h"

#include <vector>

namespace ct {

class FilterDesign {
public:
    static std::vector<float> createFilter(size_t n, ReconstructionParams::FilterType type);
    static std::vector<float> ramp(size_t n);
    static std::vector<float> sheppLogan(size_t n);
    static std::vector<float> hamming(size_t n);
};

} // namespace ct
