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
        size_t m = 0;
        if (i < n / 2) {
            m = i;
        } else {
            m = n - i;
        }
        if (m == 0) {
            continue;
        }
        const double term = 1.0 / (kPi * static_cast<double>(m));
        f[i] = static_cast<float>(-1.0 * term * term);
    }

    auto ff = FFT::forward(f);
    std::vector<float> filter(n, 0.0f);
    const int nn = static_cast<int>(n);
    #pragma omp parallel for
    for (int i = 0; i < nn; ++i) {
        filter[static_cast<size_t>(i)] = static_cast<float>(2.0 * ff[static_cast<size_t>(i)].real());
    }
    filter[0] = 0.0f;
    return filter;
}

std::vector<float> FilterDesign::sheppLogan(const size_t n) {
    auto filter = ramp(n);
    const int nn = static_cast<int>(n);
    #pragma omp parallel for
    for (int k = 1; k < nn; ++k) {
        const auto ki = static_cast<size_t>(k);
        const double freq = (ki <= n / 2) ? static_cast<double>(ki) / static_cast<double>(n)
                                          : static_cast<double>(static_cast<long long>(ki) - static_cast<long long>(n)) / static_cast<double>(n);
        const double omega = kPi * freq;
        if (std::abs(omega) > 1e-12) {
            filter[ki] = static_cast<float>(filter[ki] * (std::sin(omega) / omega));
        }
    }
    filter[0] = 0.0f;
    return filter;
}

std::vector<float> FilterDesign::hamming(const size_t n) {
    auto filter = ramp(n);
    const double W = 0.5; // Нормированная частота Найквиста

    const int nn = static_cast<int>(n);
    #pragma omp parallel for
    for (int k = 0; k < nn; ++k) {
        const auto ki = static_cast<size_t>(k);
        const double freq = (ki <= n / 2) ? static_cast<double>(ki) / static_cast<double>(n)
                                          : static_cast<double>(static_cast<long long>(ki) - static_cast<long long>(n)) / static_cast<double>(n);
        const double window = 0.54 + 0.46 * std::cos(kPi * freq / W);
        filter[ki] = static_cast<float>(filter[ki] * window);
    }
    filter[0] = 0.0f;
    return filter;
}

std::vector<float> FilterDesign::cosine(const size_t n) {
    auto filter = ramp(n);
    const int nn = static_cast<int>(n);
    #pragma omp parallel for
    for (int k = 0; k < nn; ++k) {
        const auto ki = static_cast<size_t>(k);
        const double freq = (ki <= n / 2) ? static_cast<double>(ki) / static_cast<double>(n)
                                          : static_cast<double>(static_cast<long long>(ki) - static_cast<long long>(n)) / static_cast<double>(n);
        const double window = std::cos(kPi * freq);
        filter[ki] = static_cast<float>(filter[ki] * window);
    }
    filter[0] = 0.0f;
    return filter;
}

std::vector<float> FilterDesign::hann(const size_t n) {
    auto filter = ramp(n);
    const double W = 0.5; // Нормированная частота Найквиста

    const int nn = static_cast<int>(n);
    #pragma omp parallel for
    for (int k = 0; k < nn; ++k) {
        const auto ki = static_cast<size_t>(k);
        const double freq = (ki <= n / 2) ? static_cast<double>(ki) / static_cast<double>(n)
                                          : static_cast<double>(static_cast<long long>(ki) - static_cast<long long>(n)) / static_cast<double>(n);
        const double window = 0.5 + 0.5 * std::cos(kPi * freq / W);
        filter[ki] = static_cast<float>(filter[ki] * window);
    }
    filter[0] = 0.0f;
    return filter;
}

std::vector<float> FilterDesign::bartlett(const size_t n) {
    auto filter = ramp(n);
    const double W = 0.5; // Нормированная частота Найквиста

    const int nn = static_cast<int>(n);
    #pragma omp parallel for
    for (int k = 0; k < nn; ++k) {
        const auto ki = static_cast<size_t>(k);
        const double freq = (ki <= n / 2) ? static_cast<double>(ki) / static_cast<double>(n)
                                          : static_cast<double>(static_cast<long long>(ki) - static_cast<long long>(n)) / static_cast<double>(n);
        const double window = 1.0 - std::min(1.0, std::abs(freq) / W);
        filter[ki] = static_cast<float>(filter[ki] * window);
    }
    filter[0] = 0.0f;
    return filter;
}

}
