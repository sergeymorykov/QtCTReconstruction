#include "RadonTransform.h"

#include "Utils.h"

#include <cmath>
#include <omp.h>

namespace ct {

Sinogram RadonTransform::forward(const Slice& slice, const size_t num_angles, const size_t detector_bins, float min_hu, float max_hu) {
    Sinogram sino;
    if (slice.empty() || slice.width == 0 || slice.height == 0 || num_angles == 0 || detector_bins == 0) {
        return sino;
    }

    const size_t w = slice.width;
    const size_t h = slice.height;
    const float cx = static_cast<float>(w - 1) * 0.5f;
    const float cy = static_cast<float>(h - 1) * 0.5f;
    const float detector_center = static_cast<float>(detector_bins / 2);

    float span = max_hu - min_hu;
    if (span < 1e-6f) span = 1.0f;
    const float inv_span = 1.0f / span;

    // Cache-friendly local layout: [Angle][Bin]
    Buffer2D local_sino(detector_bins, num_angles, 0.0f);

    const int na = static_cast<int>(num_angles);
    const int db = static_cast<int>(detector_bins);
    
    // Angle Batching: Process a group of angles for each pixel to maximize 
    // memory bandwidth and CPU instruction throughput (similar to backprojection).
    const int batch_size = 16;
    for (int b_start = 0; b_start < na; b_start += batch_size) {
        int b_count = std::min(batch_size, na - b_start);
        
        // Pre-calculate trigonometry and coordinate bases for the current batch
        float b_cos[16], b_sin[16], b_u_base[16];
        float* b_rows[16];
        
        for (int b = 0; b < b_count; ++b) {
            float th = utils::degToRad(180.0f / static_cast<float>(na) * static_cast<float>(b_start + b));
            b_cos[b] = std::cos(th);
            b_sin[b] = std::sin(th);
            b_u_base[b] = -cx * b_cos[b] + cy * b_sin[b] + detector_center;
            b_rows[b] = local_sino[static_cast<size_t>(b_start + b)];
        }

        // Processing the slice for this batch of angles
        for (size_t y = 0; y < h; ++y) {
            const float yy = static_cast<float>(y);
            const float* slice_ptr = &slice.data[y * w];

            // Invariant coordinates for each angle in the batch (at x=0 for this row)
            float b_u_y[16];
            for (int b = 0; b < b_count; ++b) {
                b_u_y[b] = -yy * b_sin[b] + b_u_base[b];
            }
            
            for (size_t x = 0; x < w; ++x) {
                const float v_raw = slice_ptr[x];
                if (v_raw == 0.0f) continue;

                const float v = (v_raw - min_hu) * inv_span;
                const float xx = static_cast<float>(x);

                for (int b = 0; b < b_count; ++b) {
                    const float u = xx * b_cos[b] + b_u_y[b];
                    const int i0 = static_cast<int>(u);
                    
                    if (i0 >= 0 && i0 < db - 1) {
                        const float frac = u - static_cast<float>(i0);
                        b_rows[b][i0] += v * (1.0f - frac);
                        b_rows[b][i0 + 1] += v * frac;
                    } else if (i0 == db - 1) {
                        b_rows[b][i0] += v;
                    }
                }
            }
        }
    }

    // Final result in standard layout: [Bin][Angle]
    sino.data.assign(num_angles, detector_bins, 0.0f);
    for (int a = 0; a < na; ++a) {
        const float* src_row = local_sino[static_cast<size_t>(a)];
        for (int i = 0; i < db; ++i) {
            sino.data[static_cast<size_t>(i)][static_cast<size_t>(a)] = src_row[i];
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
