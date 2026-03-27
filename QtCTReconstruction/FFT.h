#pragma once

#include "CTTypes.h"

#include <vector>

namespace ct {

class FFT {
public:
    static std::vector<Complex> forward(const std::vector<float>& input);
    static std::vector<float> inverse(const std::vector<Complex>& input);
    static std::vector<float> convolve(const std::vector<float>& x, const std::vector<float>& h, bool zero_pad = true);

private:
    static void bitReverse(std::vector<Complex>& data);
    static void fftCore(std::vector<Complex>& data, bool inverse);
    static bool isPowerOfTwo(size_t n);
    static std::vector<float> zeroPadToPowerOfTwo(const std::vector<float>& input, size_t min_size = 0);
};

} // namespace ct
