#pragma once

#include <complex>
#include <cstddef>
#include <vector>

namespace ct {

using Complex = std::complex<double>;
using Slice = std::vector<std::vector<float>>;

struct Volume {
    std::vector<Slice> data;
    std::vector<float> x_coords;
    std::vector<float> y_coords;
    std::vector<float> z_coords;

    size_t width() const { return data.empty() || data[0].empty() ? 0 : data[0][0].size(); }
    size_t height() const { return data.empty() ? 0 : data[0].size(); }
    size_t depth() const { return data.size(); }
};

struct Sinogram {
    std::vector<std::vector<float>> data;
    std::vector<float> angles_deg;
    float detector_spacing_mm = 1.0f;
};

struct ReconstructionParams {
    enum class FilterType { Ramp, SheppLogan, Hamming };

    FilterType filter = FilterType::SheppLogan;
    size_t num_angles = 180;
    bool zero_padding = true;
};

} // namespace ct
