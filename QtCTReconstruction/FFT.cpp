#include "FFT.h"

#include "Utils.h"

#include <cmath>
#include <stdexcept>

namespace ct {

std::vector<Complex> FFT::m_twiddle_forward;
std::vector<Complex> FFT::m_twiddle_inverse;
size_t FFT::m_cached_n = 0;

std::vector<Complex> FFT::forward(const std::vector<float>& input) {
    std::vector<Complex> out;
    forward(input, out);
    return out;
}

void FFT::forward(const std::vector<float>& input, std::vector<Complex>& output) {
    if (input.empty()) {
        output.clear();
        return;
    }

    size_t n = input.size();
    if (!isPowerOfTwo(n)) {
        n = utils::nextPowerOfTwo(n);
    }

    if (output.size() != n) {
        output.assign(n, Complex(0.0, 0.0));
    } else {
        std::fill(output.begin(), output.end(), Complex(0.0, 0.0));
    }

    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = Complex(static_cast<double>(input[i]), 0.0);
    }

    fftCore(output, false);
}

std::vector<float> FFT::inverse(const std::vector<Complex>& input) {
    std::vector<float> out;
    inverse(input, out);
    return out;
}

void FFT::inverse(const std::vector<Complex>& input, std::vector<float>& output) {
    if (input.empty()) {
        output.clear();
        return;
    }
    if (!isPowerOfTwo(input.size())) {
        throw std::invalid_argument("FFT inverse input size must be power of two");
    }

    const size_t n = input.size();
    std::vector<Complex> data = input;
    fftCore(data, true);

    if (output.size() != n) {
        output.resize(n);
    }
    for (size_t i = 0; i < n; ++i) {
        output[i] = static_cast<float>(data[i].real());
    }
}

std::vector<float> FFT::convolve(const std::vector<float>& x, const std::vector<float>& h, const bool zero_pad) {
    if (x.empty() || h.empty()) {
        return {};
    }

    size_t target_size = std::max(x.size(), h.size());
    if (zero_pad) {
        target_size = x.size() + h.size() - 1;
    }
    target_size = utils::nextPowerOfTwo(target_size);

    const auto x_pad = zeroPadToPowerOfTwo(x, target_size);
    const auto h_pad = zeroPadToPowerOfTwo(h, target_size);
    auto xf = forward(x_pad);
    const auto hf = forward(h_pad);

    for (size_t i = 0; i < xf.size(); ++i) {
        xf[i] *= hf[i];
    }

    auto y = inverse(xf);
    if (zero_pad) {
        y.resize(x.size() + h.size() - 1);
    } else {
        y.resize(std::max(x.size(), h.size()));
    }
    return y;
}

void FFT::bitReverse(std::vector<Complex>& data) {
    size_t j = 0;
    const size_t n = data.size();
    for (size_t i = 1; i < n; ++i) {
        size_t bit = n >> 1;
        while ((j & bit) != 0) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) {
            std::swap(data[i], data[j]);
        }
    }
}

void FFT::fftCore(std::vector<Complex>& data, const bool inverse) {
    const size_t n = data.size();
    if (n <= 1) return;
    
    bitReverse(data);

    if (m_cached_n != n) {
        m_twiddle_forward = buildTwiddleTable(n, false);
        m_twiddle_inverse = buildTwiddleTable(n, true);
        m_cached_n = n;
    }

    const auto& twiddle = inverse ? m_twiddle_inverse : m_twiddle_forward;

    for (size_t len = 2; len <= n; len <<= 1) {
        const size_t step = n / len;
        for (size_t i = 0; i < n; i += len) {
            for (size_t j = 0; j < len / 2; ++j) {
                const Complex u = data[i + j];
                const Complex v = data[i + j + len / 2] * twiddle[j * step];
                data[i + j] = u + v;
                data[i + j + len / 2] = u - v;
            }
        }
    }

    if (inverse) {
        const double inv_n = 1.0 / static_cast<double>(n);
        for (auto& v : data) {
            v *= inv_n;
        }
    }
}

std::vector<Complex> FFT::buildTwiddleTable(size_t n, bool inverse) {
    std::vector<Complex> table(n / 2);
    const double angle_base = 2.0 * 3.14159265358979323846 / static_cast<double>(n) * (inverse ? 1.0 : -1.0);
    for (size_t i = 0; i < n / 2; ++i) {
        const double angle = angle_base * static_cast<double>(i);
        table[i] = Complex(std::cos(angle), std::sin(angle));
    }
    return table;
}

bool FFT::isPowerOfTwo(const size_t n) {
    return n != 0 && (n & (n - 1)) == 0;
}

std::vector<float> FFT::zeroPadToPowerOfTwo(const std::vector<float>& input, const size_t min_size) {
    const size_t size = std::max(input.size(), min_size);
    const size_t padded_size = utils::nextPowerOfTwo(size);
    std::vector<float> out(padded_size, 0.0f);
    for (size_t i = 0; i < input.size(); ++i) {
        out[i] = input[i];
    }
    return out;
}

}
