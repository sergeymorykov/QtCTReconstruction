#pragma once

#include "CTTypes.h"

#include <vector>

namespace ct {

class FFT {
public:
    struct Plan {
        size_t n = 0;
        std::vector<Complex> twiddles;
        void prepare(size_t n, bool inverse);
    };

    static void forward(const std::vector<float>& input, std::vector<Complex>& output);
    static void inverse(const std::vector<Complex>& input, std::vector<float>& output);
    
    // Real-to-Complex FFT (returns n/2 + 1 complex points)
    static void forwardReal(const std::vector<float>& input, std::vector<Complex>& output);
    // Complex-to-Real inverse FFT (expects n/2 + 1 complex points)
    static void inverseReal(const std::vector<Complex>& input, std::vector<float>& output);

    static std::vector<float> convolve(const std::vector<float>& x, const std::vector<float>& h, bool zero_pad = true);

private:
    static void bitReverse(std::vector<Complex>& data);
    static void fftCore(std::vector<Complex>& data, bool inverse, const Plan* plan = nullptr);
    static bool isPowerOfTwo(size_t n);
    static std::vector<float> zeroPadToPowerOfTwo(const std::vector<float>& input, size_t min_size = 0);
};

} // namespace ct
