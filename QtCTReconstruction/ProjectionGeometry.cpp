#include "ProjectionGeometry.h"
#include "Utils.h"
#include <cmath>

namespace ct {

std::vector<ProjectionMatrix> ProjectionGeometry::createParallelBeamMatrices(
    const std::vector<float>& angles_deg,
    const CTGeometry& geom,
    float pixel_size_mm) {
    
    std::vector<ProjectionMatrix> matrices;
    matrices.reserve(angles_deg.size());

    float cx = (geom.nx - 1) * 0.5f;
    float cy = (geom.ny - 1) * 0.5f;
    float cz = (geom.nz - 1) * 0.5f;

    float det_cx = static_cast<float>(geom.nw / 2);
    float det_cy = static_cast<float>(geom.nh / 2);

    // Parallel beam matrix components (matching CUDABackend logic)
    // u = ( (x-cx)*cos - (y-cy)*sin ) / pixel_size + det_cx
    // v = (z - cz) / pixel_size + det_cy
    
    float inv_p = 1.0f / pixel_size_mm;

    for (float angle_deg : angles_deg) {
        float rad = utils::degToRad(angle_deg);
        float cos_a = std::cos(rad);
        float sin_a = std::sin(rad);

        ProjectionMatrix m;
        // Row 0 (u calculation)
        m.data[0][0] = cos_a * inv_p;
        m.data[0][1] = -sin_a * inv_p; 
        m.data[0][2] = 0.0f;
        m.data[0][3] = det_cx - (cx * cos_a - cy * sin_a) * inv_p;

        // Row 1 (v calculation)
        m.data[1][0] = 0.0f;
        m.data[1][1] = 0.0f;
        m.data[1][2] = inv_p;
        m.data[1][3] = det_cy - cz * inv_p;

        // Row 2 (w calculation, w=1 for parallel beam)
        m.data[2][0] = 0.0f;
        m.data[2][1] = 0.0f;
        m.data[2][2] = 0.0f;
        m.data[2][3] = 1.0f;

        matrices.push_back(m);
    }

    return matrices;
}

} // namespace ct
