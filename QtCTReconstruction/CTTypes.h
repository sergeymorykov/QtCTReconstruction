#pragma once

#include <complex>
#include <cstddef>
#include <vector>
#include <initializer_list>
#include <algorithm>
#include <stdexcept>

namespace ct {

using Complex = std::complex<double>;

// Замена std::vector<std::vector<float>> на плоский 1D массив с 2D-интерфейсом.
struct Buffer2D {
    std::vector<float> data;
    size_t width = 0;
    size_t height = 0;

    Buffer2D() = default;
    Buffer2D(size_t w, size_t h, float value = 0.0f) 
        : width(w), height(h), data(w * h, value) {}
    
    Buffer2D(std::initializer_list<std::initializer_list<float>> list) {
        height = list.size();
        width = (height > 0) ? list.begin()->size() : 0;
        data.reserve(width * height);
        for (const auto& row : list) {
            data.insert(data.end(), row.begin(), row.end());
        }
    }

    bool empty() const { return data.empty(); }
    size_t size() const { return height; } // Возвращает количество строк (height)

    void assign(size_t w, size_t h, float value = 0.0f) {
        width = w;
        height = h;
        data.assign(width * height, value);
    }
    
    // Доступ через operator[] позволяющий slice[y][x]
    float* operator[](size_t y) { return data.data() + y * width; }
    const float* operator[](size_t y) const { return data.data() + y * width; }
    
    float& at(size_t x, size_t y) { return data[y * width + x]; }
    const float& at(size_t x, size_t y) const { return data[y * width + x]; }
    
    float* rowPtr(size_t y) { return data.data() + y * width; }
    const float* rowPtr(size_t y) const { return data.data() + y * width; }

    // Итераторы для поддержки заголовочных конструкций типа for (auto row : slice)
    struct RowIterator {
        float* ptr;
        size_t stride;
        float* operator*() const { return ptr; }
        RowIterator& operator++() { ptr += stride; return *this; }
        bool operator!=(const RowIterator& other) const { return ptr != other.ptr; }
    };

    RowIterator begin() { return { data.data(), width }; }
    RowIterator end() { return { data.data() + height * width, width }; }
    
    struct ConstRowIterator {
        const float* ptr;
        size_t stride;
        const float* operator*() const { return ptr; }
        ConstRowIterator& operator++() { ptr += stride; return *this; }
        bool operator!=(const ConstRowIterator& other) const { return ptr != other.ptr; }
    };

    ConstRowIterator begin() const { return { data.data(), width }; }
    ConstRowIterator end() const { return { data.data() + height * width, width }; }
};

using Slice = Buffer2D;

struct Buffer3D {
    std::vector<float> data;
    size_t width = 0;
    size_t height = 0;
    size_t depth = 0;

    std::vector<float> x_coords;
    std::vector<float> y_coords;
    std::vector<float> z_coords;

    Buffer3D() = default;
    Buffer3D(size_t w, size_t h, size_t d, float value = 0.0f) 
        : width(w), height(h), depth(d), data(w * h * d, value) {}

    bool empty() const { return data.empty(); }

    void assign(size_t w, size_t h, size_t d, float value = 0.0f) {
        width = w;
        height = h;
        depth = d;
        data.assign(width * height * depth, value);
    }

    float& at(size_t x, size_t y, size_t z) { return data[(z * height + y) * width + x]; }
    const float& at(size_t x, size_t y, size_t z) const { return data[(z * height + y) * width + x]; }

    // Получить срез (z)
    Buffer2D getSlice(size_t z) const {
        if (z >= depth) return Buffer2D();
        Buffer2D slice(width, height);
        std::copy(data.begin() + z * width * height, data.begin() + (z + 1) * width * height, slice.data.begin());
        return slice;
    }
    
    void setSlice(size_t z, const Buffer2D& slice) {
        if (z < depth && slice.width == width && slice.height == height) {
            std::copy(slice.data.begin(), slice.data.end(), data.begin() + z * width * height);
        }
    }
};

using Volume = Buffer3D;

// Структуры для облака точек
struct Point {
    float x;
    float y;
    float z;
    float hu;
};

using PointCloud = std::vector<Point>;

struct Sinogram {
    Buffer2D data; // width = num_angles, height = detector_bins
    std::vector<float> angles_deg;
    float detector_spacing_mm = 1.0f;
    float original_min_hu = 0.0f;
    float original_max_hu = 0.0f;
};

struct ReconstructionParams {
    enum class FilterType { Ramp, SheppLogan, Hamming, Cosine, Hann, Bartlett };

    FilterType filter = FilterType::SheppLogan;
    size_t num_angles = 180;
    bool zero_padding = true;
};

} // namespace ct
