#pragma once

#include "CTTypes.h"
#include <vector>

namespace ct {

class OptimizedBackprojectionCPU {
public:
    /**
     * @brief Fully optimized volume backprojection
     * 
     * Uses:
     * - Transposed projections for cache efficiency
     * - Batch processing to minimize memory writes
     * - OpenMP SIMD for vectorization
     * - Mandatory geometric symmetry (assumes centered object)
     */
    static void reconstruct(Volume& volume,
                            const Buffer2D& projections,
                            const std::vector<ProjectionMatrix>& matrices,
                            const CTGeometry& geom,
                            int batch_size = 32);

private:
    static void transposeProjections(const Buffer2D& src, 
                                    float* dst, 
                                    int np, int nh, int nw);

    static void backprojectionBatch(Volume& volume, 
                                   const float* transposed_projs,
                                   const std::vector<ProjectionMatrix>& matrices,
                                   int total_projections,
                                   int batch_size,
                                   const CTGeometry& geom);
};

} // namespace ct
