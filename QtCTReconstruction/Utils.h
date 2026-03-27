#pragma once

#include "CTTypes.h"

#include <cstddef>
#include <string>
#include <vector>

namespace ct::utils {

std::vector<float> linspace(float start, float stop, size_t count);
size_t nextPowerOfTwo(size_t v);
float degToRad(float deg);
float clamp(float v, float lo, float hi);
float lerp(float a, float b, float t);

Slice createSlice(size_t height, size_t width, float value = 0.0f);
Slice subtract(const Slice& a, const Slice& b);
float bilinearAt(const Slice& image, float x, float y, float out_of_bounds = 0.0f);

bool ensureDirectory(const std::wstring& path);

} // namespace ct::utils
