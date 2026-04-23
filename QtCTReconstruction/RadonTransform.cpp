#include "RadonTransform.h"

#include "Utils.h"

#include <cmath>
#include <omp.h>

namespace ct {

Sinogram RadonTransform::forward(const Slice& slice, const size_t num_angles, const size_t detector_bins, float min_hu, float max_hu) {
    return forward(slice.data.data(), slice.width, slice.height, num_angles, detector_bins, min_hu, max_hu);
}

Sinogram RadonTransform::forward(const float* slice_data, size_t w, size_t h, size_t num_angles, size_t detector_bins, float min_hu, float max_hu) {
    Sinogram sino;
    if (!slice_data || w == 0 || h == 0 || num_angles == 0 || detector_bins == 0) {
        return sino;
    }

    const float cx = static_cast<float>(w - 1) * 0.5f;
    const float cy = static_cast<float>(h - 1) * 0.5f;
    const float detector_center = static_cast<float>(detector_bins / 2);

    float span = max_hu - min_hu;
    if (span < 1e-6f) span = 1.0f;
    const float inv_span = 1.0f / span;

    sino.data.assign(detector_bins, num_angles, 0.0f);

    const int na = static_cast<int>(num_angles);
    const int db = static_cast<int>(detector_bins);

    // Each thread processes one angle independently:
    // reads the full slice, writes only to sino.data[a] (no races).
    #pragma omp parallel for schedule(static)
    for (int a = 0; a < na; ++a) {
        const float th = utils::degToRad(180.0f / static_cast<float>(na) * static_cast<float>(a));
        const float cos_a = std::cos(th);
        const float sin_a = std::sin(th);
        const float u_base = -cx * cos_a + cy * sin_a + detector_center;
        float* row = sino.data[static_cast<size_t>(a)];

        for (size_t y = 0; y < h; ++y) {
            const float yy = static_cast<float>(y);
            const float u_y = -yy * sin_a + u_base;
            const float* slice_ptr = slice_data + y * w;

            for (size_t x = 0; x < w; ++x) {
                const float v_raw = slice_ptr[x];
                if (v_raw == 0.0f) continue;

                const float v = (v_raw - min_hu) * inv_span;
                const float u = static_cast<float>(x) * cos_a + u_y;
                const int i0 = static_cast<int>(u);

                if (i0 >= 0 && i0 < db - 1) {
                    const float frac = u - static_cast<float>(i0);
                    row[i0]     += v * (1.0f - frac);
                    row[i0 + 1] += v * frac;
                } else if (i0 == db - 1) {
                    row[i0] += v;
                }
            }
        }
    }

    sino.angles_deg.resize(num_angles, 0.0f);
    const float angle_step = 180.0f / static_cast<float>(num_angles);
    for (size_t a = 0; a < num_angles; ++a) {
        sino.angles_deg[a] = angle_step * static_cast<float>(a);
    }
    sino.detector_spacing_mm = 1.0f;
    sino.original_min_hu = min_hu;
    sino.original_max_hu = max_hu;

    return sino;
}

}
