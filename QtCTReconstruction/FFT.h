#pragma once

#include "CTTypes.h"

#include <vector>

namespace ct {

class FFT {
public:
    // Existing API (kept for FilterDesign and tests)
    static std::vector<Complex> forward(const std::vector<float>& input);
    static std::vector<float> inverse(const std::vector<Complex>& input);
    static std::vector<float> convolve(const std::vector<float>& x, const std::vector<float>& h, bool zero_pad = true);

    // In-place API: no heap allocations, caller owns buffers, n must be power of two.
    // forwardInPlace: fills spectrum[0..n) from input[0..n), then applies forward FFT.
    // inverseInPlace: applies inverse FFT to spectrum in-place; result is in spectrum[i].real().
    static void forwardInPlace(const float* input, Complex* spectrum, size_t n);
    static void inverseInPlace(Complex* spectrum, size_t n);

private:
    static void bitReverse(Complex* data, size_t n);
    static void fftCore(Complex* data, size_t n, bool inverse);
    static bool isPowerOfTwo(size_t n);
    static std::vector<float> zeroPadToPowerOfTwo(const std::vector<float>& input, size_t min_size = 0);
};

} // namespace ct
