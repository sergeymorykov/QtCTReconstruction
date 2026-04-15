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

// Константная память для таблиц пре-вычисленных синусов/косинусов
__constant__ float c_cos_table[2048];
__constant__ float c_sin_table[2048];

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

__global__ void radonForwardScatterKernel(
    const float* __restrict__ image,
    float* __restrict__ sino,
    int width, int height,
    int num_angles, int detector_bins,
    float cx, float cy,
    float detector_center,
    int z_slice) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float v = image[y * width + x];
    if (v == 0.0f) return;

    float xx = (float)x - cx;
    float yy = (float)y - cy;

    for (int a = 0; a < num_angles; ++a) {
        float c = c_cos_table[a];
        float s = c_sin_table[a];

        float u = xx * c - yy * s + detector_center;
        
        int i0 = (int)floorf(u);
        int i1 = i0 + 1;
        float frac = u - (float)i0;

        if (i0 >= 0 && i0 < detector_bins) {
            atomicAdd(&sino[(z_slice * detector_bins + i0) * num_angles + a], v * (1.0f - frac));
        }
        if (i1 >= 0 && i1 < detector_bins) {
            atomicAdd(&sino[(z_slice * detector_bins + i1) * num_angles + a], v * frac);
        }
    }
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
        // Чтение из Constant Memory
        float t = xx * c_cos_table[a] - yy * c_sin_table[a];
        float u = t + detector_center;
        
        // Ручная линейная интерполяция для точности float32 (вместо 9-бит в текстуре)
        // tex2D с cudaFilterModePoint возвращает значение в центре пикселя (индекс + 0.5f)
        float u_idx = u; 
        int i0 = (int)floorf(u_idx);
        int i1 = i0 + 1;
        float frac = u_idx - (float)i0;

        float v0 = 0.0f;
        float v1 = 0.0f;

        // Используем tex2D с fetch в центры (mode point)
        if (i0 >= 0 && i0 < detector_bins) {
            v0 = tex2D<float>(tex, (float)a + 0.5f, (float)i0 + 0.5f);
        }
        if (i1 >= 0 && i1 < detector_bins) {
            v1 = tex2D<float>(tex, (float)a + 0.5f, (float)i1 + 0.5f);
        }

        sum += v0 + (v1 - v0) * frac;
    }

    float scale = (3.14159265358979323846f) / (2.0f * (float)num_angles);
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
    
    if (m_d_vol_in) { cudaFree(m_d_vol_in); m_d_vol_in = nullptr; }
    if (m_d_vol_out) { cudaFree(m_d_vol_out); m_d_vol_out = nullptr; }
    if (m_d_sino) { cudaFree(m_d_sino); m_d_sino = nullptr; }
    if (m_d_spectrum) { cudaFree(m_d_spectrum); m_d_spectrum = nullptr; }

    if (m_planR2C) { cufftDestroy(m_planR2C); m_planR2C = 0; }
    if (m_planC2R) { cufftDestroy(m_planC2R); m_planC2R = 0; }
    
    m_filterSize = 0;
    m_planAngles = 0;
    m_trigAngles = 0;
    m_volSize = 0;
    m_sinoSize = 0;
    m_spectrumSize = 0;
}

void CUDABackend::ensureWorkspace(size_t w, size_t h, size_t d, size_t num_angles, size_t bins) const {
    size_t vol_req = w * h * d;
    size_t sino_req = num_angles * bins * d;

    if (m_volSize < vol_req) {
        if (m_d_vol_in) cudaFree(m_d_vol_in);
        if (m_d_vol_out) cudaFree(m_d_vol_out);
        CUDA_CHECK(cudaMalloc(&m_d_vol_in, vol_req * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&m_d_vol_out, vol_req * sizeof(float)));
        m_volSize = vol_req;
    }

    if (m_sinoSize < sino_req) {
        if (m_d_sino) cudaFree(m_d_sino);
        CUDA_CHECK(cudaMalloc(&m_d_sino, sino_req * sizeof(float)));
        m_sinoSize = sino_req;
    }

    // Буфер для спектра (FFT)
    const size_t square_bins = static_cast<size_t>(std::ceil(std::sqrt(2.0) * static_cast<double>(bins)));
    const size_t padding_factor = 8;
    const size_t projection_size_padded = std::max<size_t>(64, utils::nextPowerOfTwo(padding_factor * square_bins));
    size_t complex_size = projection_size_padded / 2 + 1;
    size_t spec_req = num_angles * complex_size;

    if (m_spectrumSize < spec_req) {
        if (m_d_spectrum) cudaFree(m_d_spectrum);
        CUDA_CHECK(cudaMalloc(&m_d_spectrum, spec_req * sizeof(cufftComplex)));
        m_spectrumSize = spec_req;
    }
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

    // Дополнительно копируем в Constant Memory (ограничение 2048 углов)
    int count = (int)std::min(n, (size_t)2048);
    cudaMemcpyToSymbol(c_cos_table, hcos.data(), count * sizeof(float));
    cudaMemcpyToSymbol(c_sin_table, hsin.data(), count * sizeof(float));

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

    // Унифицировано с CPU: целочисленное деление
    float detector_center = (float)((int)detector_bins / 2);
    float cx = (float)(w - 1) * 0.5f;
    float cy = (float)(h - 1) * 0.5f;

    sino.data.assign(num_angles, detector_bins, 0.0f);
    sino.angles_deg.resize(num_angles, 0.0f);
    float angle_step = 180.0f / (float)num_angles;
    for (size_t a = 0; a < num_angles; ++a) {
        sino.angles_deg[a] = angle_step * (float)a;
    }
    sino.detector_spacing_mm = 1.0f;

    // 1. Использование Workspace (без malloc/free)
    ensureTrigTables(sino.angles_deg);
    ensureWorkspace(w, h, 1, num_angles, detector_bins);

    // 2. Подготовка текстуры из среза (Device-to-Device если бы срез был на GPU, 
    // но тут он из CPU, так что Host-to-Device один раз)
    CUDA_CHECK(cudaMemcpy(m_d_vol_in, slice.data.data(), w * h * sizeof(float), cudaMemcpyHostToDevice));

    // Находим min/max для нормализации быстро
    thrust::device_ptr<float> dp(m_d_vol_in);
    float vmin = *thrust::min_element(dp, dp + (w * h));
    float vmax = *thrust::max_element(dp, dp + (w * h));
    sino.original_min_hu = vmin;
    sino.original_max_hu = vmax;

    float span = vmax - vmin;
    if (span < 1e-6f) span = 1.0f;
    float inv_span = 1.0f / span;
    int grid_norm = (int)((w * h + 255) / 256);
    normalizeGPUKernel<<<grid_norm, 256>>>(m_d_vol_in, w * h, vmin, inv_span);

    // 3. Прямое проецирование (Scatter алгоритм - идентично CPU)
    dim3 block(16, 16);
    dim3 grid((w + 15) / 16, (h + 15) / 16);

    // Очищаем буфер синограммы перед накоплением atomicAdd
    CUDA_CHECK(cudaMemset(m_d_sino, 0, num_angles * detector_bins * sizeof(float)));

    radonForwardScatterKernel<<<grid, block>>>(
        m_d_vol_in, m_d_sino, (int)w, (int)h, 
        (int)num_angles, (int)detector_bins, 
        cx, cy, detector_center, 0
    );

    CUDA_CHECK(cudaMemcpy(sino.data.data.data(), m_d_sino, num_angles * detector_bins * sizeof(float), cudaMemcpyDeviceToHost));

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
    ensureWorkspace(output_size, output_size, 1, num_angles, detector_bins);

    // 1. Копирование Sinogram в persistent буфер (m_d_sino)
    CUDA_CHECK(cudaMemcpy(m_d_sino, sinogram.data.data.data(), num_angles * detector_bins * sizeof(float), cudaMemcpyHostToDevice));

    // 2. Padding Sinogram na GPU (используем m_d_vol_out как временный буфер)
    float* d_sino_padded = m_d_vol_out; 
    CUDA_CHECK(cudaMemset(d_sino_padded, 0, num_angles * projection_size_padded * sizeof(float)));

    {
        dim3 blk(16, 16);
        dim3 grd((num_angles + 15) / 16, (detector_bins + 15) / 16);
        packSinogramKernel<<<grd, blk>>>(
            m_d_sino, d_sino_padded,
            (int)num_angles, (int)detector_bins,
            (int)projection_size_padded, (int)pad_before);
    }

    // 3. FFT (используем persistent m_d_spectrum)
    cufftExecR2C(m_planR2C, (cufftReal*)d_sino_padded, m_d_spectrum);

    {
        int complex_size = projection_size_padded / 2 + 1;
        dim3 blk(256, 1);
        dim3 grd((complex_size + 255) / 256, (int)num_angles);
        multiplyFilterKernel<<<grd, blk>>>(m_d_spectrum, m_d_filter, (int)projection_size_padded, (int)num_angles);
    }

    cufftExecC2R(m_planC2R, m_d_spectrum, (cufftReal*)d_sino_padded);

    // 4. Transpose and scale (результат в m_d_vol_in)
    float* d_filtered = m_d_vol_in; 
    {
        float scale = 1.0f / (float)projection_size_padded;
        dim3 blk(32, 32);
        dim3 grd(((int)num_angles + 31) / 32, ((int)square_bins + 31) / 32);
        transposeScaleKernel<<<grd, blk>>>(d_sino_padded, d_filtered, (int)num_angles, (int)projection_size_padded, (int)square_bins, scale);
    }

    // 5. Backprojection (результат в m_d_vol_out)
    cudaChannelFormatDesc chDesc = cudaCreateChannelDesc<float>();
    cudaArray_t cuArray;
    CUDA_CHECK(cudaMallocArray(&cuArray, &chDesc, num_angles, square_bins));
    CUDA_CHECK(cudaMemcpy2DToArray(cuArray, 0, 0, d_filtered, num_angles * sizeof(float), num_angles * sizeof(float), square_bins, cudaMemcpyDeviceToDevice));
    
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;
    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint; // Используем POINT и ручной lerp для точности float32
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    cudaTextureObject_t texObj = 0;
    CUDA_CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

    float* d_recon = m_d_vol_out;
    {
        dim3 bp_block(16, 16);
        dim3 bp_grid((output_size + bp_block.x - 1) / bp_block.x, (output_size + bp_block.y - 1) / bp_block.y);

        backprojectionTextureKernel<<<bp_grid, bp_block>>>(
            texObj, d_recon, 
            (int)output_size, (int)num_angles, (int)square_bins, 
            (float)(output_size / 2), (float)(square_bins / 2)
        );
    }

    // 6. Denormalize
    float span = sinogram.original_max_hu - sinogram.original_min_hu;
    if (span > 0.0f) {
        int grid_denorm = (output_size * output_size + 255) / 256;
        denormalizeGPUKernel<<<grid_denorm, 256>>>(d_recon, output_size * output_size, sinogram.original_min_hu, span);
    }

    recon.assign(output_size, output_size, 0.0f);
    CUDA_CHECK(cudaMemcpy(recon.data.data(), d_recon, output_size * output_size * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaDestroyTextureObject(texObj));
    CUDA_CHECK(cudaFreeArray(cuArray));

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

    // 1. Подготовка Workspace и параметров
    ensureTrigTables(std::vector<float>(params.num_angles)); // Будущие углы
    // На самом деле углы генерируются в computeSinogram. 
    // Для упрощения возьмем стандартный шаг 180/N.
    std::vector<float> angles(params.num_angles);
    float step = 180.0f / (float)params.num_angles;
    for(size_t i=0; i<params.num_angles; ++i) angles[i] = i * step;
    ensureTrigTables(angles);

    ensureWorkspace(width, height, depth, params.num_angles, width);

    // 2. Копируем ВЕСЬ объем на GPU один раз
    CUDA_CHECK(cudaMemcpy(m_d_vol_in, input_volume.data.data(), depth * width * height * sizeof(float), cudaMemcpyHostToDevice));

    std::vector<float> slice_min(depth, 0.0f);
    std::vector<float> slice_max(depth, 0.0f);

    // 3. Цикл по срезам внутри GPU (будет использовать Workspaces)
    for (size_t z = 0; z < depth; ++z) {
        // Мы можем оптимизировать это еще сильнее, запустив ядра на весь объем,
        // но для начала устраним malloc/free и лишние копии.

        thrust::device_ptr<float> dp(m_d_vol_in + (z * width * height));
        float vmin = *thrust::min_element(dp, dp + (width * height));
        float vmax = *thrust::max_element(dp, dp + (width * height));
        
        slice_min[z] = vmin;
        slice_max[z] = vmax;

        float span = vmax - vmin;
        if (span < 1e-6f) span = 1.0f;
        float inv_span = 1.0f / span;
        int grid_norm = (int)((width * height + 255) / 256);
        normalizeGPUKernel<<<grid_norm, 256>>>(m_d_vol_in + (z * width * height), width * height, vmin, inv_span);

        // Forward Projection (Scatter)
        dim3 block(16, 16);
        dim3 grid((width + 15) / 16, (height + 15) / 16);
        
        float cx = (float)(width - 1) * 0.5f;
        float cy = (float)(height - 1) * 0.5f;
        float detector_center = (float)((int)width / 2); // Предполагаем bins = width

        // Очищаем текущий срез синограммы
        CUDA_CHECK(cudaMemset(m_d_sino + (z * params.num_angles * width), 0, params.num_angles * width * sizeof(float)));

        radonForwardScatterKernel<<<grid, block>>>(
            m_d_vol_in + (z * width * height), 
            m_d_sino, (int)width, (int)height,
            (int)params.num_angles, (int)width, 
            cx, cy, detector_center, (int)z
        );
    }

    // 4. Batch Reconstruction (FFT и Backprojection)
    // Для краткости вызываем существующий reconstructSlice, но он теперь использует Device-Device копии.
    // На реально больших данных здесь следовало бы делать Batch cuFFT.
    
    for (size_t z = 0; z < depth; ++z) {
        Sinogram s;
        s.data.assign(params.num_angles, width, 0.0f);
        // Копируем готовую синограмму из пакетного буфера обратно для reconstructSlice (или рефакторим дальше)
        // В данном случае, чтобы не ломать логику IReconstructionBackend, копируем точечно.
        CUDA_CHECK(cudaMemcpy(s.data.data.data(), m_d_sino + (z * params.num_angles * width), params.num_angles * width * sizeof(float), cudaMemcpyDeviceToHost));
        s.angles_deg = angles;
        s.original_min_hu = slice_min[z];
        s.original_max_hu = slice_max[z];

        Buffer2D recon = reconstructSlice(s, width, params);
        out_reconstruction.setSlice(z, recon);

        if (onSliceDone) onSliceDone((int)z, recon);
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
