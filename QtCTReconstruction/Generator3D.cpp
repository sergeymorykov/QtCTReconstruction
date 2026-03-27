#include "Generator3D.h"

#include "Utils.h"

#include <algorithm>
#include <omp.h>
#include <random>

namespace ct {

Volume Generator3D::generateBrainHU(const Params& params) {
    Volume vol;
    const size_t w = params.shape[0];
    const size_t h = params.shape[1];
    const size_t d = params.shape[2];

    vol.data.assign(d, utils::createSlice(h, w, -1000.0f));
    vol.x_coords = utils::linspace(-params.brain_radius_mm, params.brain_radius_mm, w);
    vol.y_coords = utils::linspace(-params.brain_radius_mm, params.brain_radius_mm, h);
    vol.z_coords = utils::linspace(-params.brain_radius_mm, params.brain_radius_mm, d);

    std::mt19937 rng(params.seed);
    std::uniform_real_distribution<float> cdist(-0.6f * params.brain_radius_mm, 0.6f * params.brain_radius_mm);
    std::uniform_real_distribution<float> adist(3.0f, 10.0f);
    //диапазон -20.0f, 45.0f характерен для мягких тканей
    std::uniform_real_distribution<float> hdist(-20.0f, 45.0f);

    const int nd = static_cast<int>(d);
    for (size_t n = 0; n < params.num_ellipsoids; ++n) {
        const float cx = cdist(rng);
        const float cy = cdist(rng);
        const float cz = cdist(rng);
        const float ax = adist(rng);
        const float ay = adist(rng);
        const float az = adist(rng);
        const float hu = hdist(rng);

        #pragma omp parallel for schedule(static)
        for (int z = 0; z < nd; ++z) {
            const auto zi = static_cast<size_t>(z);
            const float zz = vol.z_coords[zi] - cz;
            for (size_t y = 0; y < h; ++y) {
                const float yy = vol.y_coords[y] - cy;
                for (size_t x = 0; x < w; ++x) {
                    const float xx = vol.x_coords[x] - cx;
                    const float inside = (xx * xx) / (ax * ax) + (yy * yy) / (ay * ay) + (zz * zz) / (az * az);
                    if (inside <= 1.0f) {
                        vol.data[zi][y][x] = std::max(vol.data[zi][y][x], hu);
                    }
                }
            }
        }
    }

    return vol;
}

} // namespace ct
