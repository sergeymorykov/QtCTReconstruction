#include "FilterDesign.h"

#include "FFT.h"

#include <cmath>
#include <omp.h>

namespace ct {
namespace {
constexpr double kPi = 3.14159265358979323846;
}

std::vector<float> FilterDesign::createFilter(const size_t n, const ReconstructionParams::FilterType type) {
    switch (type) {
    case ReconstructionParams::FilterType::Ramp:
        return ramp(n);
    case ReconstructionParams::FilterType::Hamming:
        return hamming(n);
    case ReconstructionParams::FilterType::Cosine:
        return cosine(n);
    case ReconstructionParams::FilterType::Hann:
        return hann(n);
    case ReconstructionParams::FilterType::Bartlett:
        return bartlett(n);
    case ReconstructionParams::FilterType::SheppLogan:
    default:
        return sheppLogan(n);
    }
}

std::vector<float> FilterDesign::ramp(const size_t n) {
    std::vector<float> f(n, 0.0f);
    f[0] = 0.25f;
    for (size_t i = 1; i < n; i += 2) {
        size_t m = (i < n / 2) ? i : n - i;
        if (m == 0) continue;
        const double term = 1.0 / (kPi * static_cast<double>(m));
        f[i] = static_cast<float>(-1.0 * term * term);
    }

    const size_t spectrum_size = n / 2 + 1;
    std::vector<Complex> ff(spectrum_size);
    FFT::forwardReal(f, ff);

    std::vector<float> filter(spectrum_size, 0.0f);
    for (size_t i = 0; i < spectrum_size; ++i) {
        filter[i] = static_cast<float>(2.0 * ff[i].real());
    }
    filter[0] = 0.0f;
    return filter;
}

std::vector<float> FilterDesign::sheppLogan(const size_t n) {
    auto filter = ramp(n);
    const size_t spectrum_size = n / 2 + 1;
    for (size_t ki = 1; ki < spectrum_size; ++ki) {
        const double freq = static_cast<double>(ki) / static_cast<double>(n);
        const double omega = kPi * freq;
        if (std::abs(omega) > 1e-12) {
            filter[ki] = static_cast<float>(filter[ki] * (std::sin(omega) / omega));
        }
    }
    return filter;
}

std::vector<float> FilterDesign::hamming(const size_t n) {
    auto filter = ramp(n);
    const size_t spectrum_size = n / 2 + 1;
    const double W = 0.5; 
    for (size_t ki = 0; ki < spectrum_size; ++ki) {
        const double freq = static_cast<double>(ki) / static_cast<double>(n);
        const double window = 0.54 + 0.46 * std::cos(kPi * freq / W);
        filter[ki] = static_cast<float>(filter[ki] * window);
    }
    return filter;
}

std::vector<float> FilterDesign::cosine(const size_t n) {
    auto filter = ramp(n);
    const size_t spectrum_size = n / 2 + 1;
    for (size_t ki = 0; ki < spectrum_size; ++ki) {
        const double freq = static_cast<double>(ki) / static_cast<double>(n);
        const double window = std::cos(kPi * freq);
        filter[ki] = static_cast<float>(filter[ki] * window);
    }
    return filter;
}

std::vector<float> FilterDesign::hann(const size_t n) {
    auto filter = ramp(n);
    const size_t spectrum_size = n / 2 + 1;
    const double W = 0.5;
    for (size_t ki = 0; ki < spectrum_size; ++ki) {
        const double freq = static_cast<double>(ki) / static_cast<double>(n);
        const double window = 0.5 + 0.5 * std::cos(kPi * freq / W);
        filter[ki] = static_cast<float>(filter[ki] * window);
    }
    return filter;
}

std::vector<float> FilterDesign::bartlett(const size_t n) {
    auto filter = ramp(n);
    const size_t spectrum_size = n / 2 + 1;
    const double W = 0.5;
    for (size_t ki = 0; ki < spectrum_size; ++ki) {
        const double freq = static_cast<double>(ki) / static_cast<double>(n);
        const double window = 1.0 - std::min(1.0, std::abs(freq) / W);
        filter[ki] = static_cast<float>(filter[ki] * window);
    }
    return filter;
}

} // namespace ct
