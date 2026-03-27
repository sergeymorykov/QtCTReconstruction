#include "Generator3D.h"

#include "Utils.h"

#include <algorithm>
#include <cmath>
#include <omp.h>
#include <random>

namespace ct {
namespace {

struct EllipsoidSpec {
    float cx = 0.0f;
    float cy = 0.0f;
    float cz = 0.0f;
    float ax = 0.0f;
    float ay = 0.0f;
    float az = 0.0f;
    float hu = -1000.0f;
};

inline float uniformFloat(std::mt19937& rng, float min_value, float max_value) {
    std::uniform_real_distribution<float> dist(min_value, max_value);
    return dist(rng);
}

inline float sampleHu(std::mt19937& rng, const Generator3D::TissueParams& tissue) {
    return uniformFloat(rng, tissue.hu_min, tissue.hu_max);
}

inline float sampleSize(std::mt19937& rng, const Generator3D::TissueParams& tissue) {
    return uniformFloat(rng, tissue.size_min_mm, tissue.size_max_mm);
}

inline EllipsoidSpec makeEllipsoidSpec(std::mt19937& rng, const Generator3D::Params& params, const Generator3D::TissueParams& tissue) {
    constexpr float kTwoPi = 6.2831853071795864769f;
    constexpr float kHalfPi = 1.5707963267948966192f;

    const float radius = params.brain_radius_mm;
    const float theta = uniformFloat(rng, 0.0f, kTwoPi);
    const float phi = std::acos(uniformFloat(rng, -1.0f, 1.0f));

    float r = 0.0f;
    if (tissue.peripheral_only) {
        const float rmin = tissue.peripheral_radius_min * radius;
        const float rmax = tissue.peripheral_radius_max * radius;
        r = uniformFloat(rng, rmin, rmax);
    } else {
        const float u = uniformFloat(rng, 0.0f, 1.0f);
        r = radius * std::cbrt(u);
    }

    const float sin_phi = std::sin(phi);
    EllipsoidSpec spec;
    spec.cx = r * sin_phi * std::cos(theta);
    spec.cy = r * sin_phi * std::sin(theta);
    spec.cz = r * std::cos(phi);

    const float base = sampleSize(rng, tissue);
    spec.ax = base;
    spec.ay = uniformFloat(rng, tissue.size_min_mm, tissue.size_max_mm);
    spec.az = uniformFloat(rng, tissue.size_min_mm, tissue.size_max_mm);
    spec.hu = sampleHu(rng, tissue);
    return spec;
}

inline void rasterizeEllipsoid(Volume& vol,
                               const EllipsoidSpec& e,
                               const std::vector<float>& x_coords,
                               const std::vector<float>& y_coords,
                               const std::vector<float>& z_coords) {
    const int nd = static_cast<int>(z_coords.size());
    const size_t h = y_coords.size();
    const size_t w = x_coords.size();

    #pragma omp parallel for schedule(static)
    for (int z = 0; z < nd; ++z) {
        const size_t zi = static_cast<size_t>(z);
        const float dz = z_coords[zi] - e.cz;
        for (size_t y = 0; y < h; ++y) {
            const float dy = y_coords[y] - e.cy;
            for (size_t x = 0; x < w; ++x) {
                const float dx = x_coords[x] - e.cx;
                if (isInsideEllipsoid(dx, dy, dz, e.ax, e.ay, e.az)) {
                    // Preserve maximum HU on overlap: bone > soft tissue > air.
                    vol.data[zi][y][x] = std::max(vol.data[zi][y][x], e.hu);
                }
            }
        }
    }
}

} // namespace

Volume Generator3D::generateBrainHU(const Params& params) {
    Volume vol;
    const size_t w = params.shape[0];
    const size_t h = params.shape[1];
    const size_t d = params.shape[2];

    // Air background: CT convention uses approximately -1000 HU for air.
    vol.data.assign(d, utils::createSlice(h, w, -1000.0f));
    vol.x_coords = utils::linspace(-params.brain_radius_mm, params.brain_radius_mm, w);
    vol.y_coords = utils::linspace(-params.brain_radius_mm, params.brain_radius_mm, h);
    vol.z_coords = utils::linspace(-params.brain_radius_mm, params.brain_radius_mm, d);

    std::mt19937 rng(params.seed);
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);

    for (size_t n = 0; n < params.num_ellipsoids; ++n) {
        const bool choose_bone = prob_dist(rng) < params.bone_tissue.probability;
        const TissueParams& tissue = choose_bone ? params.bone_tissue : params.soft_tissue;

        EllipsoidSpec e = makeEllipsoidSpec(rng, params, tissue);
        rasterizeEllipsoid(vol, e, vol.x_coords, vol.y_coords, vol.z_coords);
    }

    return vol;
}

} // namespace ct
