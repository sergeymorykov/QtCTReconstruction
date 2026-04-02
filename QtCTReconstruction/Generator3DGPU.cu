#include "Generator3DGPU.h"
#include "Utils.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <random>
#include <vector>
#include <iostream>
#include <algorithm>

namespace ct {

namespace {

struct GpuEllipsoid {
    float cx, cy, cz;
    float ax_inv_sq, ay_inv_sq, az_inv_sq;
    float hu;
};

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        } \
    } while(0)

__constant__ GpuEllipsoid c_ellipsoids[512];

__global__ void generateVolumeKernel(
    float* vol,
    int w, int h, int d,
    float x_min, float x_max,
    float y_min, float y_max,
    float z_min, float z_max,
    int num_ellipsoids) 
{
    int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int y_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int z_idx = blockIdx.z * blockDim.z + threadIdx.z;

    if (x_idx >= w || y_idx >= h || z_idx >= d) return;

    float fx = x_min + (float)x_idx * (x_max - x_min) / (float)(w - 1);
    float fy = y_min + (float)y_idx * (y_max - y_min) / (float)(h - 1);
    float fz = z_min + (float)z_idx * (z_max - z_min) / (float)(d - 1);

    float max_hu = -1000.0f;

    for (int i = 0; i < num_ellipsoids; ++i) {
        const GpuEllipsoid& e = c_ellipsoids[i];
        float dx = fx - e.cx;
        float dy = fy - e.cy;
        float dz = fz - e.cz;

        float norm = (dx * dx * e.ax_inv_sq) + (dy * dy * e.ay_inv_sq) + (dz * dz * e.az_inv_sq);
        if (norm <= 1.0f) {
            if (e.hu > max_hu) max_hu = e.hu;
        }
    }

    vol[(z_idx * h + y_idx) * w + x_idx] = max_hu;
}

// Helpers from CPU version but adapted
inline float uniformFloat(std::mt19937& rng, float min_v, float max_v) {
    std::uniform_real_distribution<float> dist(min_v, max_v);
    return dist(rng);
}

} // namespace

bool Generator3DGPU::isAvailable() const {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

Volume Generator3DGPU::generateBrainHU(const Generator3D::Params& params) {
    Volume vol;
    const size_t w = params.shape[0];
    const size_t h = params.shape[1];
    const size_t d = params.shape[2];
    
    vol.assign(w, h, d, -1000.0f);
    
    // Preparation on CPU — тот же seed и тот же порядок вызовов RNG что и в CPU-версии
    // (важно для воспроизводимости результатов)
    std::mt19937 rng(params.seed);
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f); // Создаётся ОДИН РАЗ ДО цикла!
    std::vector<GpuEllipsoid> h_ellipsoids;
    h_ellipsoids.reserve(params.num_ellipsoids);

    for (size_t n = 0; n < params.num_ellipsoids; ++n) {
        const bool choose_bone = prob_dist(rng) < params.bone_tissue.probability;
        const auto& tissue = choose_bone ? params.bone_tissue : params.soft_tissue;

        const float radius = params.brain_radius_mm;
        const float theta = uniformFloat(rng, 0.0f, 6.2831853071795864769f);
        const float phi = std::acos(uniformFloat(rng, -1.0f, 1.0f));

        float r = 0.0f;
        if (tissue.peripheral_only) {
            r = uniformFloat(rng, tissue.peripheral_radius_min * radius, tissue.peripheral_radius_max * radius);
        } else {
            r = radius * std::cbrt(uniformFloat(rng, 0.0f, 1.0f));
        }

        const float sin_phi = std::sin(phi);
        GpuEllipsoid e;
        e.cx = r * sin_phi * std::cos(theta);
        e.cy = r * sin_phi * std::sin(theta);
        e.cz = r * std::cos(phi);

        // Порядок вызовов: base=ax, потом ay, потом az — как в CPU makeEllipsoidSpec
        const float ax = uniformFloat(rng, tissue.size_min_mm, tissue.size_max_mm); // base
        const float ay = uniformFloat(rng, tissue.size_min_mm, tissue.size_max_mm);
        const float az = uniformFloat(rng, tissue.size_min_mm, tissue.size_max_mm);
        e.ax_inv_sq = 1.0f / (ax * ax);
        e.ay_inv_sq = 1.0f / (ay * ay);
        e.az_inv_sq = 1.0f / (az * az);
        e.hu = uniformFloat(rng, tissue.hu_min, tissue.hu_max);

        h_ellipsoids.push_back(e);
    }

    // CUDA allocation
    float* d_vol = nullptr;
    CUDA_CHECK(cudaMalloc(&d_vol, w * h * d * sizeof(float)));
    
    int num_ell = (int)h_ellipsoids.size();
    if (num_ell > 512) num_ell = 512; // Limit to constant memory size

    CUDA_CHECK(cudaMemcpyToSymbol(c_ellipsoids, h_ellipsoids.data(), num_ell * sizeof(GpuEllipsoid)));

    dim3 block(8, 8, 8);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y, (d + block.z - 1) / block.z);

    float range = params.brain_radius_mm;
    generateVolumeKernel<<<grid, block>>>(d_vol, (int)w, (int)h, (int)d, -range, range, -range, range, -range, range, num_ell);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(vol.data.data(), d_vol, w * h * d * sizeof(float), cudaMemcpyDeviceToHost));
    
    cudaFree(d_vol);

    // Setup coordinates (same as CPU)
    vol.x_coords = utils::linspace(-params.brain_radius_mm, params.brain_radius_mm, w);
    vol.y_coords = utils::linspace(-params.brain_radius_mm, params.brain_radius_mm, h);
    vol.z_coords = utils::linspace(-params.brain_radius_mm, params.brain_radius_mm, d);

    return vol;
}

} // namespace ct
