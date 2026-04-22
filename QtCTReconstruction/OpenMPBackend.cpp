#include "OpenMPBackend.h"

#include "OptimizedBackprojectionCPU.h"
#include "ProjectionGeometry.h"
#include "FilteredBackprojection.h"
#include "FFT.h"
#include "FilterDesign.h"
#include "Utils.h"
#include "RadonTransform.h"
#include <chrono>

#include <algorithm>
#include <limits>
#include <omp.h>

namespace ct {

Sinogram OpenMPBackend::computeSinogram(const Buffer2D& slice, size_t num_angles, size_t detector_bins) {
    if (slice.empty() || num_angles == 0 || detector_bins == 0) return {};

    float hu_min = std::numeric_limits<float>::max();
    float hu_max = std::numeric_limits<float>::lowest();

    #pragma omp parallel for reduction(min:hu_min) reduction(max:hu_max)
    for (int y = 0; y < static_cast<int>(slice.height); ++y) {
        const float* row = &slice.data[y * slice.width];
        for (size_t x = 0; x < slice.width; ++x) {
            float v = row[x];
            if (v < hu_min) hu_min = v;
            if (v > hu_max) hu_max = v;
        }
    }

    return RadonTransform::forward(slice, num_angles, detector_bins, hu_min, hu_max);
}

Buffer2D OpenMPBackend::reconstructSlice(const Sinogram& sinogram, size_t output_size, const ReconstructionParams& params) {
    Buffer2D recon_normalized = FilteredBackprojection::reconstruct(sinogram, output_size, params);
    
    if (recon_normalized.empty()) return recon_normalized;

    Buffer2D recon_hu(recon_normalized.width, recon_normalized.height, 0.0f);
    float span = sinogram.original_max_hu - sinogram.original_min_hu;
    float hu_min = sinogram.original_min_hu;

    if (span <= 0.0f) {
        recon_hu.assign(recon_normalized.width, recon_normalized.height, hu_min);
        return recon_hu;
    }

    #pragma omp parallel for
    for (int y = 0; y < static_cast<int>(recon_normalized.height); ++y) {
        for (size_t x = 0; x < recon_normalized.width; ++x) {
            recon_hu.data[y * recon_normalized.width + x] = recon_normalized.data[y * recon_normalized.width + x] * span + hu_min;
        }
    }

    return recon_hu;
}

void OpenMPBackend::reconstructVolume(const Volume& input_volume, 
                                      Volume& out_reconstruction,
                                      const ReconstructionParams& params,
                                      std::function<void(int slice_idx, const Buffer2D& recon_slice)> onSliceDone) {
    if (input_volume.empty()) return;
    
    const size_t depth = input_volume.depth;
    const size_t width = input_volume.width;
    const size_t height = input_volume.height;
    const size_t num_angles = params.num_angles;
    const size_t detector_bins = width; // Standard for this app

    out_reconstruction.assign(width, height, depth, 0.0f);
    out_reconstruction.x_coords = input_volume.x_coords;
    out_reconstruction.y_coords = input_volume.y_coords;
    out_reconstruction.z_coords = input_volume.z_coords;

    // --- Phase 1: Forward Projection & Filtering ---
    auto t_start = std::chrono::steady_clock::now();
    
    // We store all filtered projections in one buffer
    // For consistency with FilteredBackprojection, we use square_bins
    const size_t square_bins = static_cast<size_t>(std::ceil(std::sqrt(2.0) * static_cast<double>(detector_bins)));
    const size_t pad_before = (square_bins / 2) - (detector_bins / 2);
    
    // nh = depth, nw = square_bins, np = num_angles
    Buffer2D filtered_projs_all(square_bins, num_angles * depth, 0.0f);

    std::vector<float> min_hus(depth);
    std::vector<float> spans(depth);

    // Padding parameters for filtering (reusing FilteredBackprojection logic)
    const size_t padding_factor = 2;
    const size_t projection_size_padded = std::max<size_t>(64, utils::nextPowerOfTwo(padding_factor * square_bins));
    const auto filter = FilterDesign::createFilter(projection_size_padded, params.filter);

    #pragma omp parallel for schedule(dynamic)
    for (int z = 0; z < static_cast<int>(depth); ++z) {
        Buffer2D original_slice = input_volume.getSlice(z);
        Sinogram sino = computeSinogram(original_slice, num_angles, detector_bins);
        
        min_hus[z] = sino.original_min_hu;
        float s = sino.original_max_hu - sino.original_min_hu;
        spans[z] = (s <= 0.0f) ? 1.0f : s;

        // Filter this sinogram
        for (size_t a = 0; a < num_angles; ++a) {
            std::vector<float> proj_padded(projection_size_padded, 0.0f);
            for (size_t i = 0; i < detector_bins; ++i) {
                // Centered padding:
                size_t dst_idx = i + pad_before;
                if (dst_idx < projection_size_padded) {
                    // Reverted Layout: [bin][angle]
                    proj_padded[dst_idx] = sino.data[i][a];
                }
            }

            auto spectrum = FFT::forward(proj_padded);
            for (size_t k = 0; k < spectrum.size(); ++k) {
                spectrum[k] *= static_cast<double>(filter[k]);
            }
            const auto q = FFT::inverse(spectrum);
            
            // Store in global buffer at index (a * depth + z)
            float* dst_row = filtered_projs_all.rowPtr(a * depth + z);
            for (size_t i = 0; i < square_bins; ++i) {
                dst_row[i] = q[i];
            }
        }
    }

    auto t_filter = std::chrono::steady_clock::now();
    m_lastSinogramTimeMs = std::chrono::duration<double, std::milli>(t_filter - t_start).count();

    // --- Phase 2: Optimized 3D Backprojection ---
    
    CTGeometry geom;
    geom.nx = static_cast<int>(width);
    geom.ny = static_cast<int>(height);
    geom.nz = static_cast<int>(depth);
    geom.np = static_cast<int>(num_angles);
    geom.nw = static_cast<int>(square_bins); // Updated nw
    geom.nh = static_cast<int>(depth);

    // Generate angles
    std::vector<float> angles(num_angles);
    const float angle_step = 180.0f / static_cast<float>(num_angles);
    for (size_t a = 0; a < num_angles; ++a) {
        angles[a] = angle_step * static_cast<float>(a);
    }

    auto matrices = ProjectionGeometry::createParallelBeamMatrices(angles, geom);

    OptimizedBackprojectionCPU::reconstruct(out_reconstruction, filtered_projs_all, matrices, geom, 32);

    auto t_end = std::chrono::steady_clock::now();
    
    // Scale result and map back to HU values
    const float scale = 3.14159265358979323846f / (2.0f * static_cast<float>(num_angles));
    
    #pragma omp parallel for
    for (int z = 0; z < static_cast<int>(depth); ++z) {
        float slice_span = spans[z];
        float slice_min = min_hus[z];
        float combined_scale = scale * slice_span;
        
        float* slice_ptr = &out_reconstruction.data[z * width * height];
        for (size_t i = 0; i < width * height; ++i) {
            slice_ptr[i] = (slice_ptr[i] * combined_scale) + slice_min;
        }
    }

    // Call onSliceDone if needed (optional for volume reconstruction)
    if (onSliceDone) {
        for (size_t z = 0; z < depth; ++z) {
            onSliceDone(static_cast<int>(z), out_reconstruction.getSlice(z));
        }
    }
}

PointCloud OpenMPBackend::extractPointCloud(const Volume& vol, float threshold) {
    if (vol.empty()) return {};

    PointCloud cloud;
    
    #pragma omp parallel
    {
        PointCloud local_cloud;
        #pragma omp for nowait
        for (int z = 0; z < static_cast<int>(vol.depth); ++z) {
            float z_coord = vol.z_coords.empty() ? static_cast<float>(z) : vol.z_coords[z];
            for (size_t y = 0; y < vol.height; ++y) {
                float y_coord = vol.y_coords.empty() ? static_cast<float>(y) : vol.y_coords[y];
                for (size_t x = 0; x < vol.width; ++x) {
                    float x_coord = vol.x_coords.empty() ? static_cast<float>(x) : vol.x_coords[x];
                    float hu = vol.at(x, y, z);
                    if (hu > threshold) {
                        local_cloud.push_back({x_coord, y_coord, z_coord, hu});
                    }
                }
            }
        }
        
        #pragma omp critical
        {
            cloud.insert(cloud.end(), local_cloud.begin(), local_cloud.end());
        }
    }
    
    return cloud;
}

} // namespace ct
