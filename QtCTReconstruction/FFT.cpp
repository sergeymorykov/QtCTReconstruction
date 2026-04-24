#include "FFT.h"
#include "Utils.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace ct {

void FFT::Plan::prepare(size_t n, bool inverse) {
    if (this->n == n && !twiddles.empty()) return;
    this->n = n;
    twiddles.resize(n / 2);
    const float angle_base = 2.0f * 3.14159265359f / static_cast<float>(n) * (inverse ? 1.0f : -1.0f);
    for (size_t i = 0; i < n / 2; ++i) {
        const float angle = angle_base * static_cast<float>(i);
        twiddles[i] = Complex(std::cos(angle), std::sin(angle));
    }
}

void FFT::forward(const std::vector<float>& input, std::vector<Complex>& output) {
    if (input.empty()) {
        output.clear();
        return;
    }
    size_t n = utils::nextPowerOfTwo(input.size());
    output.assign(n, Complex(0.0f, 0.0f));
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = Complex(input[i], 0.0f);
    }
    Plan p;
    p.prepare(n, false);
    fftCore(output, false, &p);
}

void FFT::inverse(const std::vector<Complex>& input, std::vector<float>& output) {
    if (input.empty()) {
        output.clear();
        return;
    }
    size_t n = input.size();
    std::vector<Complex> data = input;
    Plan p;
    p.prepare(n, true);
    fftCore(data, true, &p);
    output.resize(n);
    for (size_t i = 0; i < n; ++i) {
        output[i] = data[i].real();
    }
}

void FFT::forwardReal(const std::vector<float>& input, std::vector<Complex>& output) {
    const size_t n = input.size();
    if (n == 0) { output.clear(); return; }
    if (n % 2 != 0) throw std::invalid_argument("R2C FFT requires even size");
    
    const size_t n2 = n / 2;
    std::vector<Complex> data(n2);
    for (size_t i = 0; i < n2; ++i) {
        data[i] = Complex(input[2 * i], input[2 * i + 1]);
    }

    Plan p;
    p.prepare(n2, false);
    fftCore(data, false, &p);

    output.resize(n2 + 1);
    const float angle_step = -3.14159265359f / static_cast<float>(n2);
    
    for (size_t i = 0; i <= n2 / 2; ++i) {
        const size_t j = (n2 - i) % n2;
        const Complex val_i = data[i];
        const Complex val_j = std::conj(data[j]);
        
        const Complex even = (val_i + val_j) * 0.5f;
        const Complex odd = (val_i - val_j) * Complex(0, -0.5f);
        
        const float angle = angle_step * static_cast<float>(i);
        const Complex twiddle(std::cos(angle), std::sin(angle));
        
        output[i] = even + twiddle * odd;
        if (i < n2 - i) {
            output[n2 - i] = std::conj(even - twiddle * odd);
        }
    }
}

void FFT::inverseReal(const std::vector<Complex>& input, std::vector<float>& output) {
    if (input.empty()) { output.clear(); return; }
    const size_t n2 = input.size() - 1;
    const size_t n = n2 * 2;
    
    std::vector<Complex> data(n2);
    const float angle_step = 3.14159265359f / static_cast<float>(n2);

    for (size_t i = 0; i < n2; ++i) {
        const size_t j = (n2 - i) % n2;
        // Invert the packing trick
        const Complex val_i = input[i];
        const Complex val_j = std::conj(input[j]);
        
        const Complex even = (val_i + val_j);
        const Complex odd = (val_i - val_j) * Complex(0, 1.0f);
        
        const float angle = angle_step * static_cast<float>(i);
        const Complex twiddle(std::cos(angle), std::sin(angle));
        
        data[i] = even + twiddle * odd; // This is 2 * (even + twiddle * odd)
    }

    Plan p;
    p.prepare(n2, true);
    fftCore(data, true, &p);
    
    output.resize(n);
    const float inv_2 = 0.5f; // Scale by 0.5 because we had 2* above
    for (size_t i = 0; i < n2; ++i) {
        output[2 * i] = data[i].real() * inv_2;
        output[2 * i + 1] = data[i].imag() * inv_2;
    }
}

void FFT::bitReverse(std::vector<Complex>& data) {
    const size_t n = data.size();
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(data[i], data[j]);
    }
}

void FFT::fftCore(std::vector<Complex>& data, bool inverse, const Plan* plan) {
    const size_t n = data.size();
    if (n <= 1) return;
    bitReverse(data);

    Plan local_plan;
    if (!plan) {
        local_plan.prepare(n, inverse);
        plan = &local_plan;
    }

    for (size_t len = 2; len <= n; len <<= 1) {
        const size_t step = n / len;
        for (size_t i = 0; i < n; i += len) {
            for (size_t j = 0; j < len / 2; ++j) {
                const Complex v = data[i + j + len / 2] * plan->twiddles[j * step];
                data[i + j + len / 2] = data[i + j] - v;
                data[i + j] += v;
            }
        }
    }

    if (inverse) {
        const float inv_n = 1.0f / static_cast<float>(n);
        for (auto& v : data) v *= inv_n;
    }
}

std::vector<float> FFT::convolve(const std::vector<float>& x, const std::vector<float>& h, bool zero_pad) {
    size_t n = x.size() + (zero_pad ? h.size() - 1 : 0);
    n = utils::nextPowerOfTwo(n);
    
    std::vector<float> x_pad(n, 0.0f);
    std::vector<float> h_pad(n, 0.0f);
    std::copy(x.begin(), x.end(), x_pad.begin());
    std::copy(h.begin(), h.end(), h_pad.begin());

    std::vector<Complex> xf, hf;
    forwardReal(x_pad, xf);
    forwardReal(h_pad, hf);

    for (size_t i = 0; i < xf.size(); ++i) xf[i] *= hf[i];

    std::vector<float> y;
    inverseReal(xf, y);
    y.resize(zero_pad ? x.size() + h.size() - 1 : std::max(x.size(), h.size()));
    return y;
}

bool FFT::isPowerOfTwo(size_t n) { return n > 0 && (n & (n - 1)) == 0; }
std::vector<float> FFT::zeroPadToPowerOfTwo(const std::vector<float>& input, size_t min_size) {
    size_t n = utils::nextPowerOfTwo(std::max(input.size(), min_size));
    std::vector<float> res(n, 0.0f);
    std::copy(input.begin(), input.end(), res.begin());
    return res;
}

}
