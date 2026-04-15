#include "Generator3DGPU.h"
#include "Utils.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
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

__device__ float uniformFloatGPU(curandState* state, float min_v, float max_v) {
    return curand_uniform(state) * (max_v - min_v) + min_v;
}

__global__ void generateEllipsoidsParamsKernel(
    GpuEllipsoid* ellipsoids, int num_ellipsoids, unsigned long long seed, float radius,
    Generator3D::TissueParams soft_tissue,
    Generator3D::TissueParams bone_tissue)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_ellipsoids) return;

    curandState state;
    curand_init(seed, i, 0, &state);

    float prob = curand_uniform(&state);
    bool choose_bone = prob < bone_tissue.probability;
    Generator3D::TissueParams tissue = choose_bone ? bone_tissue : soft_tissue;

    float theta = uniformFloatGPU(&state, 0.0f, 6.2831853071795864769f);
    float u_phi = uniformFloatGPU(&state, -1.0f, 1.0f);
    float phi = acosf(u_phi);

    float effective_radius = fmaxf(0.0f, radius - tissue.size_max_mm);
    float r = 0.0f;
    if (tissue.peripheral_only) {
        r = uniformFloatGPU(&state, tissue.peripheral_radius_min * effective_radius, tissue.peripheral_radius_max * effective_radius);
    } else {
        float u_r = uniformFloatGPU(&state, 0.0f, 1.0f);
        r = effective_radius * cbrtf(u_r);
    }

    float sin_phi = sinf(phi);
    float cx = r * sin_phi * cosf(theta);
    float cy = r * sin_phi * sinf(theta);
    float cz = r * cosf(phi);

    float ax = uniformFloatGPU(&state, tissue.size_min_mm, tissue.size_max_mm);
    float ay = uniformFloatGPU(&state, tissue.size_min_mm, tissue.size_max_mm);
    float az = uniformFloatGPU(&state, tissue.size_min_mm, tissue.size_max_mm);

    float hu = uniformFloatGPU(&state, tissue.hu_min, tissue.hu_max);

    ellipsoids[i].cx = cx;
    ellipsoids[i].cy = cy;
    ellipsoids[i].cz = cz;
    ellipsoids[i].ax_inv_sq = 1.0f / (ax * ax);
    ellipsoids[i].ay_inv_sq = 1.0f / (ay * ay);
    ellipsoids[i].az_inv_sq = 1.0f / (az * az);
    ellipsoids[i].hu = hu;
}

__global__ void generateVolumeKernelLarge(
    float* vol,
    int w, int h, int d,
    float x_min, float x_max,
    float y_min, float y_max,
    float z_min, float z_max,
    const GpuEllipsoid* __restrict__ ellipsoids,
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
        float dx = fx - ellipsoids[i].cx;
        float dy = fy - ellipsoids[i].cy;
        float dz = fz - ellipsoids[i].cz;

        float norm = (dx * dx * ellipsoids[i].ax_inv_sq) + 
                     (dy * dy * ellipsoids[i].ay_inv_sq) + 
                     (dz * dz * ellipsoids[i].az_inv_sq);
        
        if (norm <= 1.0f && ellipsoids[i].hu > max_hu) {
            max_hu = ellipsoids[i].hu;
        }
    }

    vol[(z_idx * h + y_idx) * w + x_idx] = max_hu;
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
    
    int num_ell = (int)params.num_ellipsoids;
    if (num_ell <= 0) return vol;

    // Allocate GPU buffer for ellipsoids
    GpuEllipsoid* d_ellipsoids = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ellipsoids, num_ell * sizeof(GpuEllipsoid)));

    // Generate ellipsoids parameters on GPU via cuRAND
    int blocks_params = (num_ell + 255) / 256;
    float range = params.brain_radius_mm;
    
    generateEllipsoidsParamsKernel<<<blocks_params, 256>>>(
        d_ellipsoids, num_ell, (unsigned long long)params.seed, range,
        params.soft_tissue, params.bone_tissue);

    // Rasterize volume using generated ellipsoids
    float* d_vol = nullptr;
    CUDA_CHECK(cudaMalloc(&d_vol, w * h * d * sizeof(float)));

    dim3 block(8, 8, 8);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y, (d + block.z - 1) / block.z);

    generateVolumeKernelLarge<<<grid, block>>>(d_vol, (int)w, (int)h, (int)d, 
                                               -range, range, -range, range, -range, range, 
                                               d_ellipsoids, num_ell);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy rasterized volume back to CPU
    CUDA_CHECK(cudaMemcpy(vol.data.data(), d_vol, w * h * d * sizeof(float), cudaMemcpyDeviceToHost));
    
    cudaFree(d_vol);
    cudaFree(d_ellipsoids);

    // Setup coordinates (same as CPU)
    vol.x_coords = utils::linspace(-params.brain_radius_mm, params.brain_radius_mm, w);
    vol.y_coords = utils::linspace(-params.brain_radius_mm, params.brain_radius_mm, h);
    vol.z_coords = utils::linspace(-params.brain_radius_mm, params.brain_radius_mm, d);

    return vol;
}

} // namespace ct
