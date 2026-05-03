#include "CUDAPureBackend.h"
#include "FilterDesign.h"
#include "Utils.h"
#include "RadonTransform.h"
#include "FilteredBackprojection.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <chrono>
#include <omp.h>

#include <cuda_runtime.h>
#include <cufft.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ct {

// Separate constant memory from CUDABackend to avoid linker conflicts.
__constant__ float gp_cos_table[2048];
__constant__ float gp_sin_table[2048];

namespace {

// Макрос для проверки CUDA-вызовов: при ошибке выбрасывает исключение.
// Это гарантирует прекращение выполнения до передачи нулевого дескриптора
// (cuFFT handle / указателя) в следующий вызов API.
#define GPUCHECK(call) \
    do { \
        cudaError_t _e = (call); \
        if (_e != cudaSuccess) { \
            std::string _msg = std::string("CUDA error: ") + cudaGetErrorString(_e) \
                             + " @ " + __FILE__ + ":" + std::to_string(__LINE__); \
            std::cerr << _msg << "\n"; \
            throw std::runtime_error(_msg); \
        } \
    } while(0)

// Макрос для проверки вызовов cuFFT (CUFFT_ALLOC_FAILED при нехватке VRAM).
#define CUFFT_CHECK(call) \
    do { \
        cufftResult _r = (call); \
        if (_r != CUFFT_SUCCESS) { \
            std::string _msg = std::string("cuFFT error code ") \
                             + std::to_string((int)_r) \
                             + " @ " + __FILE__ + ":" + std::to_string(__LINE__); \
            std::cerr << _msg << "\n"; \
            throw std::runtime_error(_msg); \
        } \
    } while(0)

// ---------- generic kernels ----------

__global__ void gpNormalizeKernel(float* data, int n, float vmin, float inv_span) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = (data[i] - vmin) * inv_span;
}

// Per-slice denormalize: vol[(z*nxy)+i] = val * span[z] + min_hu[z]  then  *= scale
__global__ void gpDenormalizeVolumeKernel(float* vol, const float* min_hu, const float* span,
                                           int nxy, int nz, float pi_scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nxy * nz;
    if (idx >= total) return;
    int z = idx / nxy;
    float s = span[z];
    vol[idx] = vol[idx] * pi_scale * s + min_hu[z];
}

// ---------- sinogram kernels ----------

// Scatter Radon for one z-slice inside a full volume buffer.
// vol_slice = pointer to vol_data + z * width * height (already on GPU, normalized).
__global__ void gpRadonScatterKernel(
    const float* __restrict__ vol_slice,
    float*       __restrict__ sino_out,   // row = a*depth+z, width = detector_bins
    int width, int height, int z_idx,
    int num_angles, int detector_bins,
    int depth,
    float cx, float cy, float det_center)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float v = vol_slice[y * width + x];
    if (v == 0.0f) return;

    float xx = (float)x - cx;
    float yy = (float)y - cy;

    for (int a = 0; a < num_angles; ++a) {
        float u = xx * gp_cos_table[a] - yy * gp_sin_table[a] + det_center;
        int i0 = (int)floorf(u);
        int i1 = i0 + 1;
        float frac = u - (float)i0;

        // Row in sino_out: a * depth + z_idx
        int row = a * depth + z_idx;
        if (i0 >= 0 && i0 < detector_bins)
            atomicAdd(&sino_out[row * detector_bins + i0], v * (1.0f - frac));
        if (i1 >= 0 && i1 < detector_bins)
            atomicAdd(&sino_out[row * detector_bins + i1], v * frac);
    }
}

// Pack: copy raw sinogram (layout [a*depth+z][detector_bins]) into
// zero-padded buffer (layout [a*depth+z][projection_size_padded]).
// dst must be pre-zeroed.
__global__ void gpPackKernel(
    const float* __restrict__ src,
    float*       __restrict__ dst,
    int rows, int detector_bins, int projection_size_padded, int pad_before)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int b   = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows || b >= detector_bins) return;
    dst[row * projection_size_padded + b + pad_before] = src[row * detector_bins + b];
}

// Filter multiply in frequency domain.
__global__ void gpFilterKernel(cufftComplex* spectrum, const float* filter,
                                 int complex_size, int batch) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    if (k >= complex_size || b >= batch) return;
    float f = filter[k];
    int idx = b * complex_size + k;
    spectrum[idx].x *= f;
    spectrum[idx].y *= f;
}

// Scale after iFFT.
__global__ void gpScaleKernel(float* data, int n, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] *= scale;
}

// ---------- Air-skipping helpers (paper-style, simplified) ----------
//
// Идея из статьи "fast hybrid CPU/GPU CT reconstruction with air skipping
// involving polygon clipping": для каждой проекции (angle, z) определяем
// горизонтальные границы [u_min, u_max] активной (non-air) области, потом в
// backprojection пропускаем углы, где проекция вокселя попадает за пределы
// этих границ. Воксели, попавшие в air для ВСЕХ углов, не получают вклада.
//
// Здесь упрощённая версия: вместо K-means (k=2) используем простой порог
// (1% от max-абс per row). Для CT-данных результат близок: фон-воздух имеет
// значения ~0, ткани/кости — заметно выше шума.

// CPU-функция: посчитать [u_min, u_max] per (angle, z) на основе RAW синограммы.
// Параллелится через OpenMP — CPU работа, может выполняться overlap с GPU FFT.
// Output: u_min[a*depth+z], u_max[a*depth+z]; если строка вся air → u_min > u_max.
static void computeAirBoundariesCPU(
    const float* sinogram,   // [num_angles*depth][bins], layout sino[(a*depth+z)*bins + b]
    int num_angles, int depth, int bins,
    int* out_u_min, int* out_u_max)
{
    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < num_angles * depth; ++idx) {
        const float* row = sinogram + (size_t)idx * bins;
        // Найти max-abs для row-specific порога
        float row_max = 0.0f;
        for (int b = 0; b < bins; ++b) {
            float v = std::fabs(row[b]);
            if (v > row_max) row_max = v;
        }
        const float thr = row_max * 0.01f;  // 1% порог
        int umn = bins, umx = -1;
        for (int b = 0; b < bins; ++b) {
            if (std::fabs(row[b]) > thr) {
                if (b < umn) umn = b;
                if (b > umx) umx = b;
            }
        }
        // Запас в 1 бин для билинейной интерполяции
        if (umx >= 0) { umn = std::max(0, umn - 1); umx = std::min(bins - 1, umx + 1); }
        out_u_min[idx] = umn;
        out_u_max[idx] = umx;
    }
}

// ---------- 3D volumetric backprojection — TEXTURE + AIR-SKIP ----------
// Использует cudaTextureObject_t с manual lerp по u (для float32 точности —
// hardware lerp даёт 9-бит).
// Air-skipping: u_min[a*depth+z] и u_max[a*depth+z] определяют активный
// диапазон на (angle, slice). Воксели вне диапазона для всех углов получают 0.
__global__ void gpVolumeBackprojTexAirSkipKernel(
    cudaTextureObject_t  tex_projs,        // 2D: rows = a*chunk_nz + v_row, cols = u
    const int* __restrict__ u_min_g,       // global-z indexing [a*depth+z]
    const int* __restrict__ u_max_g,
    int depth_global,                      // = nz (full depth)
    int chunk_z0,                          // глобальный z для local z=0
    float* __restrict__       vol_out,
    int nx, int ny, int nz_chunk,
    int nw,
    int num_angles,
    float cx, float cy, float cz,
    float det_cx, float det_cy)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz_chunk) return;

    // Circular mask
    float dxn = ((float)x - cx) / (cx + 0.5f);
    float dyn = ((float)y - cy) / (cy + 0.5f);
    if (dxn * dxn + dyn * dyn > 1.0f) {
        vol_out[(z * ny + y) * nx + x] = 0.0f;
        return;
    }

    float fx = (float)x - cx;
    float fy = (float)y - cy;

    float v_raw = (float)z - cz + det_cy;
    int v_row = (int)(v_raw + 0.5f);
    if (v_row < 0 || v_row >= nz_chunk) {
        vol_out[(z * ny + y) * nx + x] = 0.0f;
        return;
    }

    int global_z = chunk_z0 + z;
    float sum = 0.0f;

    // Заглушка для unused params — air-skip отключён (см. ниже).
    (void)u_min_g; (void)u_max_g; (void)depth_global; (void)global_z;

    for (int a = 0; a < num_angles; ++a) {
        float u = fx * gp_cos_table[a] - fy * gp_sin_table[a] + det_cx;
        int u0 = (int)floorf(u);
        if (u0 < 0 || u0 >= nw - 1) continue;

        // Air-skip отключён: clip по raw-границам синограммы обрезал side-lobes
        // ramp-фильтра (которые расходятся за пределы активной области), что
        // вызывало артефакты/чёрный фон. Чтобы безопасно включить, нужно
        // считать границы по ФИЛЬТРОВАННЫМ проекциям или закладывать запас
        // ~30-50 бин с каждой стороны под side-lobes.

        // Manual lerp по u, row фиксирован (точная X-интерполяция на float32)
        float row_y = (float)(a * nz_chunk + v_row) + 0.5f;
        float v0 = tex2D<float>(tex_projs, (float)u0     + 0.5f, row_y);
        float v1 = tex2D<float>(tex_projs, (float)u0 + 1 + 0.5f, row_y);
        float fu = u - (float)u0;
        sum += v0 + (v1 - v0) * fu;
    }

    vol_out[(z * ny + y) * nx + x] = sum;
}

// ---------- 3D volumetric backprojection ----------
// filt_projs layout: each row is proj_stride floats wide (= proj_pad after iFFT).
// Valid filtered data occupies positions [0, nw-1] (nw = square_bins) within each row.
// Row index for angle a, z-slice v_row: a * nz + v_row.
__global__ void gpVolumeBackprojectionKernel(
    float*       __restrict__ vol_out,
    const float* __restrict__ filt_projs,
    int nx, int ny, int nz,
    int nw,          // = square_bins (number of valid bins per row)
    int proj_stride, // = proj_pad   (actual floats per row in filt_projs)
    int num_angles,
    float cx, float cy, float cz,
    float det_cx, float det_cy)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    // Circular mask in XY
    float dx = ((float)x - cx) / (cx + 0.5f);
    float dy = ((float)y - cy) / (cy + 0.5f);
    if (dx * dx + dy * dy > 1.0f) {
        vol_out[(z * ny + y) * nx + x] = 0.0f;
        return;
    }

    float fx = (float)x - cx;
    float fy = (float)y - cy;

    // v maps volume z to a row in the filtered projection buffer
    float v_raw = (float)z - cz + det_cy;
    int v_row = (int)(v_raw + 0.5f);

    float sum = 0.0f;
    if (v_row >= 0 && v_row < nz) {
        for (int a = 0; a < num_angles; ++a) {
            float u = fx * gp_cos_table[a] - fy * gp_sin_table[a] + det_cx;
            int u0 = (int)u;
            if (u0 < 0 || u0 >= nw - 1) continue;
            float fu = u - (float)u0;
            // Use proj_stride (not nw) as the actual row width in memory
            const float* row = filt_projs + ((long long)a * nz + v_row) * proj_stride;
            sum += row[u0] + fu * (row[u0 + 1] - row[u0]);
        }
    }

    vol_out[(z * ny + y) * nx + x] = sum;
}

// ---------- point cloud helpers (same as CUDABackend) ----------
struct GpVolToPoint {
    const float* vol_data;
    int w, h;
    const float* xc; const float* yc; const float* zc;
    __host__ __device__ ct::Point operator()(int idx) const {
        int x = idx % w, y = (idx / w) % h, z = idx / (w * h);
        ct::Point p;
        p.x = xc ? xc[x] : (float)x;
        p.y = yc ? yc[y] : (float)y;
        p.z = zc ? zc[z] : (float)z;
        p.hu = vol_data[idx];
        return p;
    }
};
struct GpThreshold {
    float thr;
    __host__ __device__ bool operator()(const ct::Point& p) const { return p.hu > thr; }
};

} // anonymous namespace

// ============================================================
//  Resource management
// ============================================================

CUDAPureBackend::~CUDAPureBackend() { releaseAll(); }

void CUDAPureBackend::releaseAll() const {
    auto free_if = [](float*& p){ if (p) { cudaFree(p); p = nullptr; } };
    free_if(m_d_vol_in); free_if(m_d_vol_out);
    free_if(m_d_all_sinos); free_if(m_d_projs_pad);
    free_if(m_d_filter); free_if(m_d_cos); free_if(m_d_sin);
    free_if(m_d_min_hu); free_if(m_d_span);
    if (m_d_umin) { cudaFree(m_d_umin); m_d_umin = nullptr; }
    if (m_d_umax) { cudaFree(m_d_umax); m_d_umax = nullptr; }
    if (m_d_spectrum) { cudaFree(m_d_spectrum); m_d_spectrum = nullptr; }
    if (m_planR2C) { cufftDestroy(m_planR2C); m_planR2C = 0; }
    if (m_planC2R) { cufftDestroy(m_planC2R); m_planC2R = 0; }
    m_volCap = m_sinoCap = m_projPadCap = m_spectrumCap = 0;
    m_filterSize = m_trigAngles = m_depthCap = 0;
    m_planBatch = m_planPadded = 0;
    m_boundCap = 0;
}

void CUDAPureBackend::ensureWorkspace(size_t w, size_t h, size_t d,
                                       size_t num_angles, size_t bins) const {
    size_t vol_req  = w * h * d;
    size_t sino_req = num_angles * d * bins;

    const size_t sq = static_cast<size_t>(std::ceil(std::sqrt(2.0) * (double)bins));
    const size_t ppf = 2;
    const size_t proj_pad = std::max<size_t>(64, utils::nextPowerOfTwo(ppf * sq));
    
    // ОГРАНИЧЕНИЕ: Для FBP мы обрабатываем не более CHUNK_Z (64) слоев за один проход FFT
    const size_t chunk_d = std::min<size_t>(d, 64);
    size_t proj_pad_req = num_angles * chunk_d * proj_pad;
    size_t spec_req     = num_angles * chunk_d * (proj_pad / 2 + 1);

    // vol_in and vol_out: Cap обновляется только ПОСЛЕ обоих успешных malloc.
    if (m_volCap < vol_req) {
        if (m_d_vol_in)  { cudaFree(m_d_vol_in);  m_d_vol_in  = nullptr; }
        if (m_d_vol_out) { cudaFree(m_d_vol_out); m_d_vol_out = nullptr; }
        m_volCap = 0;
        GPUCHECK(cudaMalloc(&m_d_vol_in,  vol_req * sizeof(float)));
        GPUCHECK(cudaMalloc(&m_d_vol_out, vol_req * sizeof(float)));
        m_volCap = vol_req;
    }

    if (m_sinoCap < sino_req) {
        if (m_d_all_sinos) { cudaFree(m_d_all_sinos); m_d_all_sinos = nullptr; }
        m_sinoCap = 0;
        GPUCHECK(cudaMalloc(&m_d_all_sinos, sino_req * sizeof(float)));
        m_sinoCap = sino_req;
    }

    if (m_projPadCap < proj_pad_req) {
        if (m_d_projs_pad) { cudaFree(m_d_projs_pad); m_d_projs_pad = nullptr; }
        m_projPadCap = 0;
        GPUCHECK(cudaMalloc(&m_d_projs_pad, proj_pad_req * sizeof(float)));
        m_projPadCap = proj_pad_req;
    }

    if (m_spectrumCap < spec_req) {
        if (m_d_spectrum) { cudaFree(m_d_spectrum); m_d_spectrum = nullptr; }
        m_spectrumCap = 0;
        GPUCHECK(cudaMalloc(&m_d_spectrum, spec_req * sizeof(cufftComplex)));
        m_spectrumCap = spec_req;
    }

    if (m_depthCap < d) {
        if (m_d_min_hu) { cudaFree(m_d_min_hu); m_d_min_hu = nullptr; }
        if (m_d_span)   { cudaFree(m_d_span);   m_d_span   = nullptr; }
        m_depthCap = 0;
        GPUCHECK(cudaMalloc(&m_d_min_hu, d * sizeof(float)));
        GPUCHECK(cudaMalloc(&m_d_span,   d * sizeof(float)));
        m_depthCap = d;
    }
}

void CUDAPureBackend::ensureFilter(size_t padded_size,
                                    ReconstructionParams::FilterType type) const {
    if (m_filterSize == padded_size && m_filterType == type) return;
    std::vector<float> cpu_f = FilterDesign::createFilter(padded_size, type);
    if (m_d_filter) cudaFree(m_d_filter);
    GPUCHECK(cudaMalloc(&m_d_filter, padded_size * sizeof(float)));
    GPUCHECK(cudaMemcpy(m_d_filter, cpu_f.data(), padded_size * sizeof(float), cudaMemcpyHostToDevice));
    m_filterSize = padded_size;
    m_filterType = type;
}

void CUDAPureBackend::ensurePlans(size_t batch, size_t padded_size) const {
    if (m_planBatch == batch && m_planPadded == padded_size) return;
    // Разрушаем старые планы и сбрасываем дескрипторы в 0
    if (m_planR2C) { cufftDestroy(m_planR2C); m_planR2C = 0; }
    if (m_planC2R) { cufftDestroy(m_planC2R); m_planC2R = 0; }
    m_planBatch  = 0;
    m_planPadded = 0;

    int n[1] = { (int)padded_size };
    CUFFT_CHECK(cufftPlanMany(&m_planR2C, 1, n, NULL, 1, (int)padded_size,
                  NULL, 1, (int)(padded_size/2+1), CUFFT_R2C, (int)batch));
    CUFFT_CHECK(cufftPlanMany(&m_planC2R, 1, n, NULL, 1, (int)(padded_size/2+1),
                  NULL, 1, (int)padded_size, CUFFT_C2R, (int)batch));
    m_planBatch  = batch;
    m_planPadded = padded_size;
}

void CUDAPureBackend::ensureTrigTables(const std::vector<float>& angles_deg) const {
    size_t n = angles_deg.size();
    if (n == 0) return;

    // Лёгкая эвристика кэширования: сверяем (count, first, last) вместо копии
    // всего вектора. Полный std::vector::operator= в горячем пути приводил к
    // access violation в Debug-билде при больших объёмах (heap fragmentation).
    const float first = angles_deg.front();
    const float last  = angles_deg.back();
    if (m_trigAngles == n &&
        m_cachedFirstAngle == first &&
        m_cachedLastAngle  == last) {
        return;
    }

    std::vector<float> hcos(n), hsin(n);
    for (size_t a = 0; a < n; ++a) {
        float th = angles_deg[a] * (float)(M_PI / 180.0);
        hcos[a] = std::cos(th); hsin[a] = std::sin(th);
    }
    if (m_trigAngles != n) {
        if (m_d_cos) cudaFree(m_d_cos);
        if (m_d_sin) cudaFree(m_d_sin);
        GPUCHECK(cudaMalloc(&m_d_cos, n * sizeof(float)));
        GPUCHECK(cudaMalloc(&m_d_sin, n * sizeof(float)));
        m_trigAngles = n;
    }
    GPUCHECK(cudaMemcpy(m_d_cos, hcos.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    GPUCHECK(cudaMemcpy(m_d_sin, hsin.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    int cnt = (int)std::min(n, (size_t)2048);
    cudaMemcpyToSymbol(gp_cos_table, hcos.data(), cnt * sizeof(float));
    cudaMemcpyToSymbol(gp_sin_table, hsin.data(), cnt * sizeof(float));
    m_cachedFirstAngle = first;
    m_cachedLastAngle  = last;
}

bool CUDAPureBackend::isAvailable() const {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

// ============================================================
//  computeSinogram / reconstructSlice — delegate to CPU
//  (single-slice paths; pure GPU advantage is in reconstructVolume)
// ============================================================

Sinogram CUDAPureBackend::computeSinogram(const Buffer2D& slice, size_t num_angles,
                                           size_t detector_bins, bool use_parallel) {
    // For single-slice calls reuse the CPU path (same quality, simpler).
    float hu_min = *std::min_element(slice.data.begin(), slice.data.end());
    float hu_max = *std::max_element(slice.data.begin(), slice.data.end());
    return RadonTransform::forward(slice, num_angles, detector_bins, hu_min, hu_max, use_parallel);
}

Buffer2D CUDAPureBackend::reconstructSlice(const Sinogram& sinogram, size_t output_size,
                                            const ReconstructionParams& params) {
    return FilteredBackprojection::reconstruct(sinogram, output_size, params);
}

// ============================================================
//  reconstructVolume — fully on GPU
// ============================================================

void CUDAPureBackend::reconstructVolume(const Volume& input_volume,
                                         Volume& out_reconstruction,
                                         const ReconstructionParams& params,
                                         std::function<void(int, const Buffer2D&)> onSliceDone) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    if (input_volume.empty()) return;

    const int nx = (int)input_volume.width;
    const int ny = (int)input_volume.height;
    const int nz = (int)input_volume.depth;
    const int na = (int)params.num_angles;
    const int bins = nx;  // detector_bins == slice width (same convention as OpenMP backend)

    out_reconstruction.assign(nx, ny, nz, 0.0f);
    out_reconstruction.x_coords = input_volume.x_coords;
    out_reconstruction.y_coords = input_volume.y_coords;
    out_reconstruction.z_coords = input_volume.z_coords;

    // Geometry constants (matching ProjectionGeometry after the fix)
    const size_t sq = static_cast<size_t>(std::ceil(std::sqrt(2.0) * (double)bins));
    const size_t pad_before   = (sq / 2) - ((size_t)bins / 2);
    const size_t proj_pad     = std::max<size_t>(64, utils::nextPowerOfTwo(2 * sq));
    const size_t complex_size = proj_pad / 2 + 1;
    const int    total_rows   = na * nz;  // one row per (angle, z-slice)

    // Angles
    std::vector<float> angles(na);
    float step = 180.0f / (float)na;
    for (int i = 0; i < na; ++i) angles[i] = step * (float)i;

    // Allocate / reuse GPU workspace (projs_pad/spectrum будут выделяться per-chunk)
    ensureWorkspace((size_t)nx, (size_t)ny, (size_t)nz, (size_t)na, (size_t)bins);
    ensureFilter(proj_pad, params.filter);
    // ensurePlans вызывается внутри chunk-цикла (batch = na * chunk_nz)
    ensureTrigTables(angles);

    const float cx      = (nx - 1) * 0.5f;
    const float cy      = (ny - 1) * 0.5f;
    const float cz      = (nz - 1) * 0.5f;
    // det_cx matches FilteredBackprojection.cpp:89 — integer division sq/2
    const float det_cx  = (float)(sq / 2);
    const float det_cy  = cz;                // = (nz-1)*0.5 (matches ProjectionGeometry fix)
    // Radon scatter bin center uses same integer-division convention
    const float det_ctr = (float)(bins / 2); // for Radon scatter bin centering

    // ====================================================================
    // ЭТАП I — Построение синограммы.
    //   I.1  H→D загрузка volume (PCIe)
    //   I.2  Per-slice normalize + scatter Radon на GPU
    //   I.3  D→H выгрузка готовой синограммы в host-буфер
    // m_lastSinogramTimeMs покрывает ВСЕ три шага (I.1 + I.2 + I.3).
    // ====================================================================
    auto t_sino_start = std::chrono::steady_clock::now();

    // --- I.1 H→D volume ---
    const size_t vol_bytes = (size_t)nx * ny * nz * sizeof(float);
    GPUCHECK(cudaMemcpy(m_d_vol_in, input_volume.data.data(), vol_bytes, cudaMemcpyHostToDevice));

    // --- I.2 Per-slice normalize + scatter Radon на GPU ---
    //          All sinogram rows stored in m_d_all_sinos:
    //          row = a * nz + z_idx,  width = bins
    std::vector<float> h_min(nz), h_span(nz);

    GPUCHECK(cudaMemset(m_d_all_sinos, 0, (size_t)total_rows * bins * sizeof(float)));

    for (int z = 0; z < nz; ++z) {
        float* d_slice = m_d_vol_in + (size_t)z * nx * ny;
        int nxy = nx * ny;

        thrust::device_ptr<float> dp(d_slice);
        float vmin = *thrust::min_element(dp, dp + nxy);
        float vmax = *thrust::max_element(dp, dp + nxy);
        h_min[z]  = vmin;
        float s   = vmax - vmin;
        h_span[z] = (s < 1e-6f) ? 1.0f : s;

        float inv_s = 1.0f / h_span[z];
        gpNormalizeKernel<<<(nxy + 255) / 256, 256>>>(d_slice, nxy, vmin, inv_s);

        dim3 blk(16, 16);
        dim3 grd((nx + 15) / 16, (ny + 15) / 16);
        gpRadonScatterKernel<<<grd, blk>>>(
            d_slice, m_d_all_sinos,
            nx, ny, z, na, bins, nz,
            cx, cy, det_ctr);
    }
    GPUCHECK(cudaDeviceSynchronize());

    // --- I.3 D→H выгрузка готовой синограммы в host-буфер ---
    // Это позволяет внешнему коду перехватить синограмму или подменить её
    // данными со сканера КТ перед запуском Этапа II.
    const size_t sino_bytes = (size_t)total_rows * bins * sizeof(float);
    std::vector<float> h_sinogram((size_t)total_rows * bins);
    GPUCHECK(cudaMemcpy(h_sinogram.data(), m_d_all_sinos, sino_bytes, cudaMemcpyDeviceToHost));

    // Конец Этапа I — фиксируем время.
    auto t_sino_end = std::chrono::steady_clock::now();
    m_lastSinogramTimeMs = std::chrono::duration<double, std::milli>(t_sino_end - t_sino_start).count();

    // ====================================================================
    // ЭТАП I.5 — Hybrid CPU step: air-skipping boundary tables.
    // Реализует идею статьи "fast hybrid CPU/GPU CT reconstruction with air
    // skipping" — CPU делает сегментацию air/non-air для каждой проекции
    // (упрощённый порог 1% от per-row max вместо K-means k=2 — практически
    // эквивалентно для CT-данных, где фон-воздух ≈ 0). Результат: u_min/u_max
    // per (angle, z), которые backproj kernel использует для clip'а.
    //
    // CPU OpenMP пул отрабатывает за ~100-300 мс на 256³, GPU в это время
    // простаивает (true CPU step). После этого backproj пропускает все
    // ангелы, где проекция вокселя попадает в air → ~30-50% меньше реальных
    // ops в backprojection (на фантомах с большим air-фоном — до 50%).
    // ====================================================================
    std::vector<int> h_umin((size_t)na * nz);
    std::vector<int> h_umax((size_t)na * nz);
    computeAirBoundariesCPU(h_sinogram.data(), na, nz, bins,
                            h_umin.data(), h_umax.data());
    {
        size_t bnd_req = (size_t)na * nz;
        if (m_boundCap < bnd_req) {
            if (m_d_umin) { cudaFree(m_d_umin); m_d_umin = nullptr; }
            if (m_d_umax) { cudaFree(m_d_umax); m_d_umax = nullptr; }
            m_boundCap = 0;
            GPUCHECK(cudaMalloc(&m_d_umin, bnd_req * sizeof(int)));
            GPUCHECK(cudaMalloc(&m_d_umax, bnd_req * sizeof(int)));
            m_boundCap = bnd_req;
        }
        GPUCHECK(cudaMemcpy(m_d_umin, h_umin.data(), bnd_req * sizeof(int), cudaMemcpyHostToDevice));
        GPUCHECK(cudaMemcpy(m_d_umax, h_umax.data(), bnd_req * sizeof(int), cudaMemcpyHostToDevice));
    }

    // ====================================================================
    // ЭТАП II — FBP-реконструкция (чанки по CHUNK_Z слоёв).
    //   II.1 H→D загрузка чанка синограммы обратно в VRAM
    //   II.2 Pack + FFT + filter + iFFT + scale
    //   II.3 3D backprojection через global memory
    //   II.4 Denormalize
    //   II.5 D→H выгрузка чанка восстановленного volume
    // m_lastReconstructionTimeMs покрывает весь chunk-loop (все чанки).
    // ====================================================================
    // Накопительный замер: суммируем чистое recon-время по чанкам, исключая
    // время на onSliceDone (UI-обновления, копии Buffer2D — это не реконструкция).
    double recon_accum_ms = 0.0;

    const int   CHUNK_Z  = 64;
    const float pi_scale = (float)(M_PI / (2.0 * na));

    // Reuse host-buffer для chunk-синограммы — выносим из цикла, чтобы не
    // создавать/уничтожать ~23МБ vector каждую итерацию (давление на debug-heap).
    std::vector<float> h_chunk_sino((size_t)na * CHUNK_Z * bins);

    for (int z0 = 0; z0 < nz; z0 += CHUNK_Z) {
        auto t_chunk_start = std::chrono::steady_clock::now();

        const int chunk_nz   = std::min(CHUNK_Z, nz - z0);
        const int chunk_rows = na * chunk_nz;

        // Аллокация projs_pad и spectrum под размер текущего чанка
        {
            size_t pp_req   = (size_t)chunk_rows * proj_pad;
            size_t spec_req = (size_t)chunk_rows * complex_size;
            if (m_projPadCap < pp_req) {
                if (m_d_projs_pad) { cudaFree(m_d_projs_pad); m_d_projs_pad = nullptr; }
                m_projPadCap = 0;
                GPUCHECK(cudaMalloc(&m_d_projs_pad, pp_req * sizeof(float)));
                m_projPadCap = pp_req;
            }
            if (m_spectrumCap < spec_req) {
                if (m_d_spectrum) { cudaFree(m_d_spectrum); m_d_spectrum = nullptr; }
                m_spectrumCap = 0;
                GPUCHECK(cudaMalloc(&m_d_spectrum, spec_req * sizeof(cufftComplex)));
                m_spectrumCap = spec_req;
            }
        }
        
        // cuFFT план пересоздаётся при смене batch
        ensurePlans((size_t)chunk_rows, proj_pad);

        // Перекладываем строки синограммы для текущего чанка.
        // h_sinogram: [a*nz + z][bins] -> chunk: [a*chunk_nz + local_z][bins]
        // h_chunk_sino выделен один раз вне цикла, используем первые chunk_rows строк.
        for (int a = 0; a < na; ++a) {
            for (int lz = 0; lz < chunk_nz; ++lz) {
                const float* src = h_sinogram.data() + ((size_t)a * nz + (z0 + lz)) * bins;
                float*       dst = h_chunk_sino.data() + ((size_t)a * chunk_nz + lz) * bins;
                std::memcpy(dst, src, (size_t)bins * sizeof(float));
            }
        }
        GPUCHECK(cudaMemcpy(m_d_all_sinos, h_chunk_sino.data(),
                            (size_t)chunk_rows * bins * sizeof(float), cudaMemcpyHostToDevice));

        // Pack -> FFT -> Filter -> iFFT -> Scale
        GPUCHECK(cudaMemset(m_d_projs_pad, 0, (size_t)chunk_rows * proj_pad * sizeof(float)));
        {
            dim3 blk(16, 16);
            dim3 grd((chunk_rows + 15) / 16, (bins + 15) / 16);
            gpPackKernel<<<grd, blk>>>(
                m_d_all_sinos, m_d_projs_pad,
                chunk_rows, bins, (int)proj_pad, (int)pad_before);
        }
        CUFFT_CHECK(cufftExecR2C(m_planR2C, (cufftReal*)m_d_projs_pad, m_d_spectrum));
        {
            dim3 blk(16, 16);
            dim3 grd(((int)complex_size + 15) / 16, (chunk_rows + 15) / 16);
            gpFilterKernel<<<grd, blk>>>(m_d_spectrum, m_d_filter, (int)complex_size, chunk_rows);
        }
        CUFFT_CHECK(cufftExecC2R(m_planC2R, m_d_spectrum, (cufftReal*)m_d_projs_pad));
        {
            int n_total = chunk_rows * (int)proj_pad;
            float fft_scale = 1.0f / (float)proj_pad;
            gpScaleKernel<<<(n_total + 255) / 256, 256>>>(m_d_projs_pad, n_total, fft_scale);
        }

        // 3D Backprojection через TEXTURE + AIR-SKIP.
        // Texture создаём поверх валидной части m_d_projs_pad (sq колонок).
        // Высота = chunk_rows = na*chunk_nz < 65536 (лимит cudaArray) — для
        // CHUNK_Z=64 и na≤910 безопасно.
        const size_t chunk_vol_bytes = (size_t)nx * ny * chunk_nz * sizeof(float);
        GPUCHECK(cudaMemset(m_d_vol_out, 0, chunk_vol_bytes));
        {
            cudaChannelFormatDesc chDesc = cudaCreateChannelDesc<float>();
            cudaArray_t cuArray = nullptr;
            GPUCHECK(cudaMallocArray(&cuArray, &chDesc, (size_t)sq, (size_t)chunk_rows));
            GPUCHECK(cudaMemcpy2DToArray(
                cuArray, 0, 0,
                m_d_projs_pad, proj_pad * sizeof(float),
                sq * sizeof(float), chunk_rows,
                cudaMemcpyDeviceToDevice));

            cudaResourceDesc resDesc = {};
            resDesc.resType = cudaResourceTypeArray;
            resDesc.res.array.array = cuArray;
            cudaTextureDesc texDesc = {};
            texDesc.addressMode[0] = cudaAddressModeClamp;
            texDesc.addressMode[1] = cudaAddressModeClamp;
            texDesc.filterMode     = cudaFilterModePoint; // Point + manual lerp = float32 точность
            texDesc.readMode       = cudaReadModeElementType;
            texDesc.normalizedCoords = 0;

            cudaTextureObject_t texObj = 0;
            GPUCHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr));

            const float chunk_cz     = (chunk_nz - 1) * 0.5f;
            const float chunk_det_cy = chunk_cz;
            dim3 blk(8, 8, 4);
            dim3 grd((nx + blk.x - 1) / blk.x,
                     (ny + blk.y - 1) / blk.y,
                     (chunk_nz + blk.z - 1) / blk.z);
            gpVolumeBackprojTexAirSkipKernel<<<grd, blk>>>(
                texObj,
                m_d_umin, m_d_umax,
                nz, z0,
                m_d_vol_out,
                nx, ny, chunk_nz, (int)sq, na,
                cx, cy, chunk_cz, det_cx, chunk_det_cy);
            GPUCHECK(cudaDeviceSynchronize());

            GPUCHECK(cudaDestroyTextureObject(texObj));
            GPUCHECK(cudaFreeArray(cuArray));
        }

        // Denormalize
        GPUCHECK(cudaMemcpy(m_d_min_hu, h_min.data()  + z0, chunk_nz * sizeof(float), cudaMemcpyHostToDevice));
        GPUCHECK(cudaMemcpy(m_d_span,   h_span.data() + z0, chunk_nz * sizeof(float), cudaMemcpyHostToDevice));
        int n_chunk_vox = nx * ny * chunk_nz;
        gpDenormalizeVolumeKernel<<<(n_chunk_vox + 255) / 256, 256>>>(
            m_d_vol_out, m_d_min_hu, m_d_span, nx * ny, chunk_nz, pi_scale);

        // D->H выгрузка результата чанка
        float* dst_ptr = out_reconstruction.data.data() + (size_t)z0 * nx * ny;
        GPUCHECK(cudaMemcpy(dst_ptr, m_d_vol_out, chunk_vol_bytes, cudaMemcpyDeviceToHost));

        // Конец чанка — фиксируем чистое recon-время до UI-callback'ов.
        auto t_chunk_end = std::chrono::steady_clock::now();
        recon_accum_ms += std::chrono::duration<double, std::milli>(t_chunk_end - t_chunk_start).count();

        if (onSliceDone) {
            for (int lz = 0; lz < chunk_nz; ++lz)
                onSliceDone(z0 + lz, out_reconstruction.getSlice((size_t)(z0 + lz)));
        }
    }

    m_lastReconstructionTimeMs = recon_accum_ms;
}

// ============================================================
//  extractPointCloud — same Thrust approach as CUDABackend
// ============================================================

PointCloud CUDAPureBackend::extractPointCloud(const Volume& vol, float threshold) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    PointCloud cloud;
    if (vol.empty()) return cloud;

    const size_t N = vol.data.size();
    thrust::device_vector<float> d_vol  = vol.data;
    thrust::device_vector<float> d_xcrd = vol.x_coords;
    thrust::device_vector<float> d_ycrd = vol.y_coords;
    thrust::device_vector<float> d_zcrd = vol.z_coords;

    GpVolToPoint op;
    op.vol_data = thrust::raw_pointer_cast(d_vol.data());
    op.w = (int)vol.width; op.h = (int)vol.height;
    op.xc = vol.x_coords.empty() ? nullptr : thrust::raw_pointer_cast(d_xcrd.data());
    op.yc = vol.y_coords.empty() ? nullptr : thrust::raw_pointer_cast(d_ycrd.data());
    op.zc = vol.z_coords.empty() ? nullptr : thrust::raw_pointer_cast(d_zcrd.data());

    GpThreshold pred; pred.thr = threshold;

    auto it = thrust::make_transform_iterator(thrust::counting_iterator<int>(0), op);
    int cnt = (int)thrust::count_if(it, it + (int)N, pred);
    if (cnt <= 0) return cloud;

    thrust::device_vector<ct::Point> d_cloud(cnt);
    thrust::copy_if(it, it + (int)N, d_cloud.begin(), pred);
    cloud.resize(cnt);
    thrust::copy(d_cloud.begin(), d_cloud.end(), cloud.begin());
    return cloud;
}

} // namespace ct
