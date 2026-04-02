#include "OpenMPBackend.h"

#include "FilteredBackprojection.h"
#include "RadonTransform.h"

namespace ct {

Sinogram OpenMPBackend::computeSinogram(const Buffer2D& slice, size_t num_angles, size_t detector_bins) {
    return RadonTransform::forward(slice, num_angles, detector_bins);
}

Buffer2D OpenMPBackend::reconstructSlice(const Sinogram& sinogram, size_t output_size, const ReconstructionParams& params) {
    return FilteredBackprojection::reconstruct(sinogram, output_size, params);
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
        // Normalize for Radon Transform mapping
        // Logic will need to match Controller if we handle min/max globally or locally
        // ... well actually controller handles min/max scaling before passing?
        // OpenMP backend itself just processes directly. The controller might do scaling.
        // I'll keep the backend strictly mathematical.
        
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
