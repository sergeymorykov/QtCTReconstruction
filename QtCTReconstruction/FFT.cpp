#include "FFT.h"

#include "Utils.h"

#include <cmath>
#include <stdexcept>

namespace ct {

namespace {
constexpr float kPi = 3.14159265358979323846f;
}

// --- In-place API (no allocations) ---

void FFT::forwardInPlace(const float* input, Complex* spectrum, const size_t n) {
    for (size_t i = 0; i < n; ++i)
        spectrum[i] = Complex(input[i], 0.0f);
    fftCore(spectrum, n, false);
}

void FFT::inverseInPlace(Complex* spectrum, const size_t n) {
    fftCore(spectrum, n, true);
}

// --- Existing vector-based API (uses in-place internally) ---

std::vector<Complex> FFT::forward(const std::vector<float>& input) {
    if (input.empty()) return {};

    std::vector<float> padded = input;
    if (!isPowerOfTwo(padded.size()))
        padded = zeroPadToPowerOfTwo(input);

    const size_t n = padded.size();
    std::vector<Complex> data(n);
    forwardInPlace(padded.data(), data.data(), n);
    return data;
}

std::vector<float> FFT::inverse(const std::vector<Complex>& input) {
    if (input.empty()) return {};
    if (!isPowerOfTwo(input.size()))
        throw std::invalid_argument("FFT inverse input size must be power of two");

    std::vector<Complex> data = input;
    fftCore(data.data(), data.size(), true);

    std::vector<float> out(data.size());
    for (size_t i = 0; i < data.size(); ++i)
        out[i] = data[i].real();
    return out;
}

std::vector<float> FFT::convolve(const std::vector<float>& x, const std::vector<float>& h, const bool zero_pad) {
    if (x.empty() || h.empty()) return {};

    size_t target_size = std::max(x.size(), h.size());
    if (zero_pad) target_size = x.size() + h.size() - 1;
    target_size = utils::nextPowerOfTwo(target_size);

    const auto x_pad = zeroPadToPowerOfTwo(x, target_size);
    const auto h_pad = zeroPadToPowerOfTwo(h, target_size);
    auto xf = forward(x_pad);
    const auto hf = forward(h_pad);

    for (size_t i = 0; i < xf.size(); ++i)
        xf[i] *= hf[i];

    auto y = inverse(xf);
    if (zero_pad)
        y.resize(x.size() + h.size() - 1);
    else
        y.resize(std::max(x.size(), h.size()));
    return y;
}

// --- Private helpers ---

void FFT::bitReverse(Complex* data, const size_t n) {
    size_t j = 0;
    for (size_t i = 1; i < n; ++i) {
        size_t bit = n >> 1;
        while ((j & bit) != 0) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) std::swap(data[i], data[j]);
    }
}

void FFT::fftCore(Complex* data, const size_t n, const bool inverse) {
    bitReverse(data, n);

    for (size_t len = 2; len <= n; len <<= 1) {
        const float angle = (inverse ? 1.0f : -1.0f) * 2.0f * kPi / static_cast<float>(len);
        const Complex wlen(std::cos(angle), std::sin(angle));
        for (size_t i = 0; i < n; i += len) {
            Complex w(1.0f, 0.0f);
            for (size_t j = 0; j < len / 2; ++j) {
                const Complex u = data[i + j];
                const Complex v = data[i + j + len / 2] * w;
                data[i + j]          = u + v;
                data[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }

    if (inverse) {
        const float inv_n = 1.0f / static_cast<float>(n);
        for (size_t i = 0; i < n; ++i)
            data[i] *= inv_n;
    }
}

bool FFT::isPowerOfTwo(const size_t n) {
    return n != 0 && (n & (n - 1)) == 0;
}

std::vector<float> FFT::zeroPadToPowerOfTwo(const std::vector<float>& input, const size_t min_size) {
    const size_t size = std::max(input.size(), min_size);
    const size_t padded_size = utils::nextPowerOfTwo(size);
    std::vector<float> out(padded_size, 0.0f);
    for (size_t i = 0; i < input.size(); ++i)
        out[i] = input[i];
    return out;
}

}
