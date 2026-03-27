#include "Utils.h"

#include <Windows.h>

#include <algorithm>
#include <cmath>
#include <omp.h>
#include <stdexcept>

namespace ct::utils {

std::vector<float> linspace(const float start, const float stop, const size_t count) {
    if (count == 0) {
        return {};
    }
    if (count == 1) {
        return {start};
    }

    std::vector<float> values(count, 0.0f);
    const float step = (stop - start) / static_cast<float>(count - 1);
    for (size_t i = 0; i < count; ++i) {
        values[i] = start + step * static_cast<float>(i);
    }
    return values;
}

size_t nextPowerOfTwo(size_t v) {
    if (v <= 1) {
        return 1;
    }
    --v;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
#if defined(_WIN64)
    v |= v >> 32;
#endif
    return v + 1;
}

float degToRad(const float deg) {
    return deg * 3.14159265358979323846f / 180.0f;
}

float clamp(const float v, const float lo, const float hi) {
    return (v < lo) ? lo : ((v > hi) ? hi : v);
}

float lerp(const float a, const float b, const float t) {
    return a + (b - a) * t;
}

Slice createSlice(const size_t height, const size_t width, const float value) {
    return Slice(height, std::vector<float>(width, value));
}

Slice subtract(const Slice& a, const Slice& b) {
    if (a.size() != b.size() || (!a.empty() && a[0].size() != b[0].size())) {
        throw std::invalid_argument("Slice size mismatch in subtract");
    }
    Slice out = a;
    const int rows = static_cast<int>(a.size());
    #pragma omp parallel for
    for (int y = 0; y < rows; ++y) {
        const auto yi = static_cast<size_t>(y);
        for (size_t x = 0; x < a[yi].size(); ++x) {
            out[yi][x] = a[yi][x] - b[yi][x];
        }
    }
    return out;
}

float bilinearAt(const Slice& image, const float x, const float y, const float out_of_bounds) {
    if (image.empty() || image[0].empty()) {
        return out_of_bounds;
    }

    const int width = static_cast<int>(image[0].size());
    const int height = static_cast<int>(image.size());

    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;

    if (x0 < 0 || y0 < 0 || x1 >= width || y1 >= height) {
        return out_of_bounds;
    }

    const float tx = x - static_cast<float>(x0);
    const float ty = y - static_cast<float>(y0);

    const float v00 = image[y0][x0];
    const float v10 = image[y0][x1];
    const float v01 = image[y1][x0];
    const float v11 = image[y1][x1];

    const float v0 = lerp(v00, v10, tx);
    const float v1 = lerp(v01, v11, tx);
    return lerp(v0, v1, ty);
}

bool ensureDirectory(const std::wstring& path) {
    if (path.empty()) {
        return false;
    }

    const DWORD attrs = GetFileAttributesW(path.c_str());
    if (attrs != INVALID_FILE_ATTRIBUTES && (attrs & FILE_ATTRIBUTE_DIRECTORY) != 0) {
        return true;
    }

    return CreateDirectoryW(path.c_str(), nullptr) != FALSE || GetLastError() == ERROR_ALREADY_EXISTS;
}

}
