#include "OpenMPBackend.h"

#include "FilteredBackprojection.h"
#include "RadonTransform.h"

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
        for (size_t x = 0; x < slice.width; ++x) {
            float v = slice.data[y * slice.width + x];
            if (v < hu_min) hu_min = v;
            if (v > hu_max) hu_max = v;
        }
    }

    Buffer2D normalized(slice.width, slice.height, 0.0f);
    float span = hu_max - hu_min;
    if (span < 1e-6f) span = 1.0f;
    float inv_span = 1.0f / span;

    #pragma omp parallel for
    for (int y = 0; y < static_cast<int>(slice.height); ++y) {
        for (size_t x = 0; x < slice.width; ++x) {
            normalized.data[y * slice.width + x] = (slice.data[y * slice.width + x] - hu_min) * inv_span;
        }
    }

    Sinogram sino = RadonTransform::forward(normalized, num_angles, detector_bins);
    sino.original_min_hu = hu_min;
    sino.original_max_hu = hu_max;

    return sino;
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
    
    out_reconstruction.assign(width, height, depth, 0.0f);
    out_reconstruction.x_coords = input_volume.x_coords;
    out_reconstruction.y_coords = input_volume.y_coords;
    out_reconstruction.z_coords = input_volume.z_coords;

    for (size_t z = 0; z < depth; ++z) {
        Buffer2D original_slice = input_volume.getSlice(z);
        Sinogram sino = computeSinogram(original_slice, params.num_angles, width);
        Buffer2D recon = reconstructSlice(sino, width, params);
        
        out_reconstruction.setSlice(z, recon);
        
        if (onSliceDone) {
            onSliceDone(static_cast<int>(z), recon);
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
