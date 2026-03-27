#include "RadonTransform.h"

#include "Utils.h"

#include <cmath>
#include <omp.h>

namespace ct {

Sinogram RadonTransform::forward(const Slice& slice, const size_t num_angles, const size_t detector_bins) {
    Sinogram sino;
    if (slice.empty() || slice[0].empty() || num_angles == 0 || detector_bins == 0) {
        return sino;
    }

    const size_t h = slice.size();
    const size_t w = slice[0].size();
    const float cx = static_cast<float>(w - 1) * 0.5f;
    const float cy = static_cast<float>(h - 1) * 0.5f;
    const float detector_center = static_cast<float>(detector_bins / 2);

    sino.data.assign(detector_bins, std::vector<float>(num_angles, 0.0f));
    sino.angles_deg.resize(num_angles, 0.0f);
    const float angle_step = 180.0f / static_cast<float>(num_angles);
    for (size_t a = 0; a < num_angles; ++a) {
        sino.angles_deg[a] = angle_step * static_cast<float>(a);
    }
    sino.detector_spacing_mm = 1.0f;

    const int na = static_cast<int>(num_angles);
    #pragma omp parallel for schedule(dynamic)
    for (int a = 0; a < na; ++a) {
        const float th = utils::degToRad(sino.angles_deg[static_cast<size_t>(a)]);
        const float c = std::cos(th);
        const float s = std::sin(th);

        for (size_t y = 0; y < h; ++y) {
            const float yy = static_cast<float>(y) - cy;
            for (size_t x = 0; x < w; ++x) {
                const float xx = static_cast<float>(x) - cx;
                const float t = xx * c - yy * s;
                const float u = t + detector_center;

                const int i0 = static_cast<int>(std::floor(u));
                const int i1 = i0 + 1;
                const float frac = u - static_cast<float>(i0);
                const float v = slice[y][x];

                if (i0 >= 0 && i0 < static_cast<int>(detector_bins)) {
                    sino.data[static_cast<size_t>(i0)][static_cast<size_t>(a)] += v * (1.0f - frac);
                }
                if (i1 >= 0 && i1 < static_cast<int>(detector_bins)) {
                    sino.data[static_cast<size_t>(i1)][static_cast<size_t>(a)] += v * frac;
                }
            }
        }
    }

    return sino;
}

}
