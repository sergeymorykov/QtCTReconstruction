#pragma once

#include "CTTypes.h"
#include <vector>

namespace ct {

class ProjectionGeometry {
public:
    /**
     * @brief Generates projection matrices for Parallel Beam geometry
     * 
     * @param angles_deg Vector of projection angles in degrees
     * @param geom CT geometry parameters
     * @param pixel_size_mm Detector pixel size (assumed same as voxel size for now)
     * @return std::vector<ProjectionMatrix> 
     */
    static std::vector<ProjectionMatrix> createParallelBeamMatrices(
        const std::vector<float>& angles_deg,
        const CTGeometry& geom,
        float pixel_size_mm = 1.0f);
};

} // namespace ct
