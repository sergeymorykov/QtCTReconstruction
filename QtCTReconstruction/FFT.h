#pragma once

#include "CTTypes.h"

#include <vector>

namespace ct {

class FFT {
public:
    static std::vector<Complex> forward(const std::vector<float>& input);
    static void forward(const std::vector<float>& input, std::vector<Complex>& output);
    static std::vector<float> inverse(const std::vector<Complex>& input);
    static void inverse(const std::vector<Complex>& input, std::vector<float>& output);
    static std::vector<float> convolve(const std::vector<float>& x, const std::vector<float>& h, bool zero_pad = true);

private:
    static void bitReverse(std::vector<Complex>& data);
    static void fftCore(std::vector<Complex>& data, bool inverse);
    static bool isPowerOfTwo(size_t n);
    static std::vector<float> zeroPadToPowerOfTwo(const std::vector<float>& input, size_t min_size = 0);
    static std::vector<Complex> buildTwiddleTable(size_t n, bool inverse);
    static std::vector<Complex> m_twiddle_forward;
    static std::vector<Complex> m_twiddle_inverse;
    static size_t m_cached_n;
};

} // namespace ct
