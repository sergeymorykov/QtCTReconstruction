#include "CUDABackend.h"

#include "FilterDesign.h"
#include "Utils.h"

#include <cmath>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <cufft.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ct {

namespace {

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        } \
    } while(0)

// Нормализация in-place на GPU
__global__ void normalizeGPUKernel(float* data, int n, float vmin, float inv_span) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = (data[i] - vmin) * inv_span;
}

// Денормализация in-place на GPU
__global__ void denormalizeGPUKernel(float* data, int n, float vmin, float span) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = data[i] * span + vmin;
}

__global__ void packSinogramKernel(
    const float* __restrict__ sino_src,
    float*       __restrict__ sino_dst,
    int num_angles, int detector_bins, int projection_size_padded, int pad_before)
{
    int angle = blockIdx.x * blockDim.x + threadIdx.x;
    int bin   = blockIdx.y * blockDim.y + threadIdx.y;
    if (angle >= num_angles || bin >= detector_bins) return;

    float val = sino_src[bin * num_angles + angle];
    sino_dst[angle * projection_size_padded + (bin + pad_before)] = val;
}

__global__ void transposeScaleKernel(
    const float* __restrict__ src,
    float*       __restrict__ dst,
    int num_angles, int projection_size_padded, int square_bins, float scale)
{
    __shared__ float tile[32][33]; 
    int ax = blockIdx.x * 32 + threadIdx.x;
    int ay = blockIdx.y * 32 + threadIdx.y;

    if (ax < num_angles && ay < square_bins)
        tile[threadIdx.y][threadIdx.x] = src[ax * projection_size_padded + ay] * scale;
    __syncthreads();

    int tx = blockIdx.y * 32 + threadIdx.x;
    int ty = blockIdx.x * 32 + threadIdx.y;
    if (tx < square_bins && ty < num_angles)
        dst[tx * num_angles + ty] = tile[threadIdx.x][threadIdx.y];
}

__global__ void radonForwardKernel(
    const float* __restrict__ slice,
    float* __restrict__ sino,
    const float* __restrict__ angles_deg,
    int width, int height,
    int num_angles, int detector_bins,
    float cx, float cy, float detector_center) 
{
    int angle_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int bin_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (angle_idx >= num_angles || bin_idx >= detector_bins) return;

    float th = angles_deg[angle_idx] * (M_PI / 180.0f);
    float c = cosf(th);
    float s = sinf(th);

    float sum = 0.0f;
    float t = (float)bin_idx - detector_center;
    
    float ray_step = 1.0f; 
    float p_min = -max(width, height) * 0.75f;
    float p_max = max(width, height) * 0.75f;
    
    for (float p = p_min; p <= p_max; p += ray_step) {
        float xx = t * c - p * s;
        float yy = -t * s - p * c;

        float x = xx + cx;
        float y = yy + cy;

        int ix = (int)floorf(x);
        int iy = (int)floorf(y);

        if (ix >= 0 && ix < width - 1 && iy >= 0 && iy < height - 1) {
            float fx = x - ix;
            float fy = y - iy;

            float v00 = slice[iy * width + ix];
            float v10 = slice[iy * width + ix + 1];
            float v01 = slice[(iy + 1) * width + ix];
            float v11 = slice[(iy + 1) * width + ix + 1];

            float v = (1.0f - fx) * (1.0f - fy) * v00 + 
                      fx * (1.0f - fy) * v10 + 
                      (1.0f - fx) * fy * v01 + 
                      fx * fy * v11;
            
            sum += v * ray_step;
        }
    }

    sino[bin_idx * num_angles + angle_idx] = sum;
}

__global__ void multiplyFilterKernel(
    cufftComplex* spectrum,
    const float* filter,
    int pad_size,
    int num_angles) 
{
    int num_elements = (pad_size / 2 + 1);
    
    int elem_x = blockIdx.x * blockDim.x + threadIdx.x; 
    int angle_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (elem_x >= num_elements || angle_idx >= num_angles) return;

    int idx = angle_idx * num_elements + elem_x;
    float f = filter[elem_x]; 
    
    spectrum[idx].x *= f;
    spectrum[idx].y *= f;
}

__global__ void backprojectionTextureKernel(
    cudaTextureObject_t tex,
    float* __restrict__ recon,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    int output_size,
    int num_angles,
    int detector_bins,
    float radius,
    float detector_center)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= output_size || y >= output_size) return;

    float xx = (float)x - radius;
    float yy = (float)y - radius;

    if (xx * xx + yy * yy > radius * radius) {
        recon[y * output_size + x] = 0.0f;
        return;
    }

    float sum = 0.0f;
    for (int a = 0; a < num_angles; ++a) {
        float t = xx * cos_table[a] - yy * sin_table[a];
        float u = t + detector_center;
        sum += tex2D<float>(tex, (float)a + 0.5f, u + 0.5f);
    }

    float scale = (M_PI) / (2.0f * num_angles);
    recon[y * output_size + x] = sum * scale;
}

struct VolToPoint {
    const float* vol_data;
    int w, h, d;
    const float* x_coords;
    const float* y_coords;
    const float* z_coords;

    __host__ __device__
    ct::Point operator()(int idx) const {
        int x = idx % w;
        int y = (idx / w) % h;
        int z = idx / (w * h);
        ct::Point p;
        p.x = x_coords ? x_coords[x] : (float)x;
        p.y = y_coords ? y_coords[y] : (float)y;
        p.z = z_coords ? z_coords[z] : (float)z;
        p.hu = vol_data[idx];
        return p;
    }
};

struct ThresholdPredicate {
    float threshold;
    __host__ __device__ bool operator()(const ct::Point& p) const {
        return p.hu > threshold;
    }
};

} // anonymous namespace

CUDABackend::~CUDABackend() {
    releaseCache();
}

void CUDABackend::releaseCache() const {
    if (m_d_filter) { cudaFree(m_d_filter); m_d_filter = nullptr; }
    if (m_d_cos) { cudaFree(m_d_cos); m_d_cos = nullptr; }
    if (m_d_sin) { cudaFree(m_d_sin); m_d_sin = nullptr; }
    if (m_planR2C) { cufftDestroy(m_planR2C); m_planR2C = 0; }
    if (m_planC2R) { cufftDestroy(m_planC2R); m_planC2R = 0; }
    m_filterSize = 0;
    m_planAngles = 0;
    m_trigAngles = 0;
}

void CUDABackend::ensureFilter(size_t padded_size, ReconstructionParams::FilterType type) const {
    if (m_filterSize == padded_size && m_filterType == type) return;
    std::vector<float> cpu_filter = FilterDesign::createFilter(padded_size, type);
    if (m_d_filter) cudaFree(m_d_filter);
    CUDA_CHECK(cudaMalloc(&m_d_filter, padded_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(m_d_filter, cpu_filter.data(), padded_size * sizeof(float), cudaMemcpyHostToDevice));
    m_filterSize = padded_size;
    m_filterType = type;
}

void CUDABackend::ensurePlans(size_t num_angles, size_t padded_size) const {
    if (m_planAngles == num_angles && m_planPadded == padded_size) return;
    if (m_planR2C) cufftDestroy(m_planR2C);
    if (m_planC2R) cufftDestroy(m_planC2R);
    int n[1] = { (int)padded_size };
    cufftPlanMany(&m_planR2C, 1, n, NULL, 1, padded_size, NULL, 1, padded_size, CUFFT_R2C, (int)num_angles);
    cufftPlanMany(&m_planC2R, 1, n, NULL, 1, padded_size, NULL, 1, padded_size, CUFFT_C2R, (int)num_angles);
    m_planAngles = num_angles;
    m_planPadded = padded_size;
}

void CUDABackend::ensureTrigTables(const std::vector<float>& angles_deg) const {
    if (m_cachedAnglesDeg == angles_deg) return;
    size_t n = angles_deg.size();
    std::vector<float> hcos(n), hsin(n);
    for (size_t a = 0; a < n; ++a) {
        float th = angles_deg[a] * (float)(M_PI / 180.0);
        hcos[a] = std::cos(th);
        hsin[a] = std::sin(th);
    }
    if (m_trigAngles != n) {
        if (m_d_cos) cudaFree(m_d_cos);
        if (m_d_sin) cudaFree(m_d_sin);
        CUDA_CHECK(cudaMalloc(&m_d_cos, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&m_d_sin, n * sizeof(float)));
    }
    CUDA_CHECK(cudaMemcpy(m_d_cos, hcos.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(m_d_sin, hsin.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    m_trigAngles = n;
    m_cachedAnglesDeg = angles_deg;
}

bool CUDABackend::isAvailable() const {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return err == cudaSuccess && count > 0;
}

Sinogram CUDABackend::computeSinogram(const Buffer2D& slice, size_t num_angles, size_t detector_bins) {
    Sinogram sino;
    if (slice.empty() || num_angles == 0 || detector_bins == 0) return sino;

    size_t w = slice.width;
    size_t h = slice.height;

    float cx = (float)(w - 1) * 0.5f;
    float cy = (float)(h - 1) * 0.5f;
    float detector_center = (float)detector_bins / 2.0f;

    sino.data.assign(num_angles, detector_bins, 0.0f);
    sino.angles_deg.resize(num_angles, 0.0f);
    float angle_step = 180.0f / (float)num_angles;
    for (size_t a = 0; a < num_angles; ++a) {
        sino.angles_deg[a] = angle_step * (float)a;
    }
    sino.detector_spacing_mm = 1.0f;

    float* d_slice = nullptr;
    float* d_sino = nullptr;
    float* d_angles = nullptr;

    CUDA_CHECK(cudaMalloc(&d_slice, w * h * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sino, num_angles * detector_bins * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_angles, num_angles * sizeof(float)));

    // Копирование сырого среза
    CUDA_CHECK(cudaMemcpy(d_slice, slice.data.data(), w * h * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_angles, sino.angles_deg.data(), num_angles * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_sino, 0, num_angles * detector_bins * sizeof(float)));

    // Нормализация на GPU
    thrust::device_ptr<float> dp(d_slice);
    float vmin = *thrust::min_element(dp, dp + (w * h));
    float vmax = *thrust::max_element(dp, dp + (w * h));

    sino.original_min_hu = vmin;
    sino.original_max_hu = vmax;

    float span = vmax - vmin;
    if (span < 1e-6f) span = 1.0f;
    float inv_span = 1.0f / span;
    int grid_norm = (int)((w * h + 255) / 256);
    normalizeGPUKernel<<<grid_norm, 256>>>(d_slice, w * h, vmin, inv_span);

    // Прямое проецирование
    dim3 block(16, 16);
    dim3 grid((num_angles + block.x - 1) / block.x, (detector_bins + block.y - 1) / block.y);

    radonForwardKernel<<<grid, block>>>(d_slice, d_sino, d_angles, w, h, num_angles, detector_bins, cx, cy, detector_center);

    CUDA_CHECK(cudaMemcpy(sino.data.data.data(), d_sino, num_angles * detector_bins * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_slice);
    cudaFree(d_sino);
    cudaFree(d_angles);

    return sino;
}

Buffer2D CUDABackend::reconstructSlice(const Sinogram& sinogram, size_t output_size, const ReconstructionParams& params) {
    Buffer2D recon;
    if (sinogram.data.empty() || output_size == 0) return recon;

    const size_t num_angles = sinogram.data.width;
    const size_t detector_bins = sinogram.data.height;

    const size_t square_bins = static_cast<size_t>(std::ceil(std::sqrt(2.0) * static_cast<double>(detector_bins)));
    const size_t pad_before = (square_bins / 2) - (detector_bins / 2);
    const size_t padding_factor = 8;
    const size_t projection_size_padded = std::max<size_t>(64, utils::nextPowerOfTwo(padding_factor * square_bins));

    ensureFilter(projection_size_padded, params.filter);
    ensurePlans(num_angles, projection_size_padded);
    ensureTrigTables(sinogram.angles_deg);

    // Копирование Sino
    float* d_sino_src = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sino_src, num_angles * detector_bins * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_sino_src, sinogram.data.data.data(), num_angles * detector_bins * sizeof(float), cudaMemcpyHostToDevice));

    // Padding Sino na GPU
    float* d_sino_padded = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sino_padded, num_angles * projection_size_padded * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_sino_padded, 0, num_angles * projection_size_padded * sizeof(float)));

    {
        dim3 blk(16, 16);
        dim3 grd((num_angles + 15) / 16, (detector_bins + 15) / 16);
        packSinogramKernel<<<grd, blk>>>(
            d_sino_src, d_sino_padded,
            (int)num_angles, (int)detector_bins,
            (int)projection_size_padded, (int)pad_before);
    }
    cudaFree(d_sino_src);

    int complex_size = projection_size_padded / 2 + 1;
    cufftComplex* d_spectrum = nullptr;
    CUDA_CHECK(cudaMalloc(&d_spectrum, num_angles * complex_size * sizeof(cufftComplex)));

    cufftExecR2C(m_planR2C, (cufftReal*)d_sino_padded, d_spectrum);

    {
        dim3 blk(256, 1);
        dim3 grd((complex_size + 255) / 256, (int)num_angles);
        multiplyFilterKernel<<<grd, blk>>>(d_spectrum, m_d_filter, (int)projection_size_padded, (int)num_angles);
    }

    cufftExecC2R(m_planC2R, d_spectrum, (cufftReal*)d_sino_padded);
    cudaFree(d_spectrum);

    // Transpose and scale
    float* d_filtered = nullptr;
    CUDA_CHECK(cudaMalloc(&d_filtered, square_bins * num_angles * sizeof(float)));
    {
        float scale = 1.0f / (float)projection_size_padded;
        dim3 blk(32, 32);
        dim3 grd(((int)num_angles + 31) / 32, ((int)square_bins + 31) / 32);
        transposeScaleKernel<<<grd, blk>>>(d_sino_padded, d_filtered, (int)num_angles, (int)projection_size_padded, (int)square_bins, scale);
    }
    cudaFree(d_sino_padded);

    std::vector<float> cpu_filtered(square_bins * num_angles);
    CUDA_CHECK(cudaMemcpy(cpu_filtered.data(), d_filtered, cpu_filtered.size() * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_filtered);

    cudaChannelFormatDesc chDesc = cudaCreateChannelDesc<float>();
    cudaArray_t cuArray;
    CUDA_CHECK(cudaMallocArray(&cuArray, &chDesc, num_angles, square_bins));
    CUDA_CHECK(cudaMemcpy2DToArray(cuArray, 0, 0, cpu_filtered.data(), num_angles * sizeof(float), num_angles * sizeof(float), square_bins, cudaMemcpyHostToDevice));

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    cudaTextureObject_t texObj = 0;
    CUDA_CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

    float* d_recon = nullptr;
    CUDA_CHECK(cudaMalloc(&d_recon, output_size * output_size * sizeof(float)));

    {
        dim3 bp_block(16, 16);
        dim3 bp_grid((output_size + bp_block.x - 1) / bp_block.x, (output_size + bp_block.y - 1) / bp_block.y);

        backprojectionTextureKernel<<<bp_grid, bp_block>>>(
            texObj, d_recon, m_d_cos, m_d_sin, 
            (int)output_size, (int)num_angles, (int)detector_bins, 
            (float)output_size / 2.0f, (float)square_bins / 2.0f
        );
    }

    // Денормализация на GPU
    float span = sinogram.original_max_hu - sinogram.original_min_hu;
    if (span > 0.0f) {
        int grid_denorm = (output_size * output_size + 255) / 256;
        denormalizeGPUKernel<<<grid_denorm, 256>>>(d_recon, output_size * output_size, sinogram.original_min_hu, span);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    recon.assign(output_size, output_size, 0.0f);
    CUDA_CHECK(cudaMemcpy(recon.data.data(), d_recon, output_size * output_size * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaDestroyTextureObject(texObj));
    CUDA_CHECK(cudaFreeArray(cuArray));
    cudaFree(d_recon);

    return recon;
}

void CUDABackend::reconstructVolume(const Volume& input_volume, 
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

PointCloud CUDABackend::extractPointCloud(const Volume& vol, float threshold) {
    PointCloud cloud;
    if (vol.empty()) return cloud;

    const size_t N = vol.data.size();
    
    thrust::device_vector<float> d_vol_data = vol.data;
    thrust::device_vector<float> d_x_coords = vol.x_coords;
    thrust::device_vector<float> d_y_coords = vol.y_coords;
    thrust::device_vector<float> d_z_coords = vol.z_coords;

    VolToPoint to_point_op;
    to_point_op.vol_data = thrust::raw_pointer_cast(d_vol_data.data());
    to_point_op.w = vol.width;
    to_point_op.h = vol.height;
    to_point_op.d = vol.depth;
    to_point_op.x_coords = vol.x_coords.empty() ? nullptr : thrust::raw_pointer_cast(d_x_coords.data());
    to_point_op.y_coords = vol.y_coords.empty() ? nullptr : thrust::raw_pointer_cast(d_y_coords.data());
    to_point_op.z_coords = vol.z_coords.empty() ? nullptr : thrust::raw_pointer_cast(d_z_coords.data());

    thrust::counting_iterator<int> iter(0);
    auto transform_iter = thrust::make_transform_iterator(iter, to_point_op);

    ThresholdPredicate pred;
    pred.threshold = threshold;

    int count = thrust::count_if(transform_iter, transform_iter + N, pred);
    if (count <= 0) return cloud;

    thrust::device_vector<ct::Point> d_cloud(count);
    thrust::copy_if(transform_iter, transform_iter + N, d_cloud.begin(), pred);

    cloud.resize(count);
    thrust::copy(d_cloud.begin(), d_cloud.end(), cloud.begin());

    return cloud;
}

} // namespace ct
