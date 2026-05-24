// HybridBackend.cu — реализация гибридного CPU+GPU FBP-конвейера.
// Подробное описание архитектуры см. HybridBackend.h.

#include "HybridBackend.h"
#include "FilterDesign.h"
#include "Utils.h"
#include "RadonTransform.h"
#include "FilteredBackprojection.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

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
#include <thrust/system/cuda/execution_policy.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ct {

// Отдельная константная память (избегает linker-конфликта с CUDABackend).
__constant__ float hb_cos_table[2048];
__constant__ float hb_sin_table[2048];

namespace {

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

// ============================================================
//  KERNELS
// ============================================================

__global__ void hbNormalizeKernel(float* __restrict__ data, int n, float vmin, float inv_span) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = (data[i] - vmin) * inv_span;
}

// Pure-GPU нормализация: vmin/vmax читаются из device-памяти (результат async-thrust на стриме).
// Это позволяет всему Этапу I быть async — никаких host-side sync на per-slice минмаксе.
__global__ void hbNormalizeFromDevicePairKernel(float* __restrict__ data, int n,
                                                const float* __restrict__ d_vminmax) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float vmin = d_vminmax[0];
    float vmax = d_vminmax[1];
    float span = vmax - vmin;
    if (span < 1e-6f) span = 1.0f;
    data[i] = (data[i] - vmin) / span;
}

// Конвертирует (vmin, vmax)[nz] → min[nz], span[nz] для последующего denormalize в Этапе II.
// Запускается один раз после всех Radon-кернелов в Этапе I.
__global__ void hbMinMaxToMinSpanKernel(const float* __restrict__ minmax,
                                         float* __restrict__ d_min,
                                         float* __restrict__ d_span,
                                         int nz) {
    int z = blockIdx.x * blockDim.x + threadIdx.x;
    if (z >= nz) return;
    float vmin = minmax[2 * z + 0];
    float vmax = minmax[2 * z + 1];
    float span = vmax - vmin;
    if (span < 1e-6f) span = 1.0f;
    d_min[z]  = vmin;
    d_span[z] = span;
}

// ---------- Gather Radon (без atomicAdd) ----------
// Поток = (bin, angle), интегрирует ОДИН луч через изображение bilinear sampling'ом.
// Каждый поток пишет в одну ячейку синограммы → coalesced writes, ноль атомиков.
// Аналог CUDABackend::radonForwardGatherKernel, но с Hybrid-layout:
//   sino[(a * depth + z_slice) * detector_bins + bin]
// (в CUDABackend: sino[(z_slice * num_angles + a) * detector_bins + bin])
__global__ void hbRadonGatherKernel(
    const float* __restrict__ image,
    float*       __restrict__ sino,
    int width, int height,
    int num_angles, int detector_bins,
    float cx, float cy,
    float detector_center,
    int z_slice, int depth)
{
    int bin = blockIdx.x * blockDim.x + threadIdx.x;
    int a   = blockIdx.y * blockDim.y + threadIdx.y;
    if (bin >= detector_bins || a >= num_angles) return;

    float c = hb_cos_table[a];
    float s = hb_sin_table[a];
    float u_signed = (float)bin - detector_center;

    // Базовая точка луча и направление, перпендикулярное проекции.
    // Согласовано со scatter-формулой: u = (x-cx)*c - (y-cy)*s + det_center.
    float bx = cx + u_signed * c;
    float by = cy - u_signed * s;
    float dx = s;
    float dy = c;

    float half = 0.5f * sqrtf((float)(width * width + height * height));
    int n_samples = (int)(2.0f * half) + 1;

    float sum = 0.0f;
    float t = -half;
    #pragma unroll 4
    for (int i = 0; i < n_samples; ++i, t += 1.0f) {
        float x = bx + t * dx;
        float y = by + t * dy;

        int x0 = (int)floorf(x);
        int y0 = (int)floorf(y);
        int x1 = x0 + 1;
        int y1 = y0 + 1;
        float fx = x - (float)x0;
        float fy = y - (float)y0;

        float v00 = (x0 >= 0 && x0 < width  && y0 >= 0 && y0 < height) ? image[y0 * width + x0] : 0.0f;
        float v10 = (x1 >= 0 && x1 < width  && y0 >= 0 && y0 < height) ? image[y0 * width + x1] : 0.0f;
        float v01 = (x0 >= 0 && x0 < width  && y1 >= 0 && y1 < height) ? image[y1 * width + x0] : 0.0f;
        float v11 = (x1 >= 0 && x1 < width  && y1 >= 0 && y1 < height) ? image[y1 * width + x1] : 0.0f;

        sum += (1.0f - fx) * (1.0f - fy) * v00
             +         fx  * (1.0f - fy) * v10
             + (1.0f - fx) *         fy  * v01
             +         fx  *         fy  * v11;
    }
    // Hybrid layout: row = a * depth + z_slice, width = detector_bins
    sino[((size_t)a * depth + z_slice) * detector_bins + bin] = sum;
}

// vol[(z*nxy)+i] = val * pi_scale * span[z] + min_hu[z]
__global__ void hbDenormalizeVolumeKernel(float* __restrict__ vol,
                                          const float* __restrict__ min_hu,
                                          const float* __restrict__ span,
                                          int nxy, int nz, float pi_scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nxy * nz;
    if (idx >= total) return;
    int z = idx / nxy;
    vol[idx] = vol[idx] * pi_scale * span[z] + min_hu[z];
}

// Scatter Radon для одного z-слоя (полный том нормализован per-slice до вызова).
__global__ void hbRadonScatterKernel(
    const float* __restrict__ vol_slice,
    float*       __restrict__ sino_out,   // row = a*depth+z, width = bins
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

    #pragma unroll 4
    for (int a = 0; a < num_angles; ++a) {
        float u = xx * hb_cos_table[a] - yy * hb_sin_table[a] + det_center;
        int i0 = (int)floorf(u);
        int i1 = i0 + 1;
        float frac = u - (float)i0;

        int row = a * depth + z_idx;
        if (i0 >= 0 && i0 < detector_bins)
            atomicAdd(&sino_out[row * detector_bins + i0], v * (1.0f - frac));
        if (i1 >= 0 && i1 < detector_bins)
            atomicAdd(&sino_out[row * detector_bins + i1], v * frac);
    }
}

__global__ void hbPackKernel(
    const float* __restrict__ src,
    float*       __restrict__ dst,
    int rows, int detector_bins, int projection_size_padded, int pad_before)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int b   = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows || b >= detector_bins) return;
    dst[row * projection_size_padded + b + pad_before] = src[row * detector_bins + b];
}

// [P1] Объединённый gather+pack одним проходом.
//   src layout: all_sinos[a * nz + z][bins]   (полный, на весь объём)
//   dst layout: pad_out[a * chunk_nz + lz][proj_pad]
//   Заменяет na отдельных cudaMemcpy2DAsync вызовов (~2-3 сек overhead на 8 чанков).
//   Block по (bins, na*chunk_nz): coalesced reads/writes в плотный pad.
__global__ void hbGatherPackKernel(
    const float* __restrict__ all_sinos,
    float*       __restrict__ pad_out,    // должен быть предварительно занулён
    int na, int nz, int bins,
    int chunk_z0, int chunk_nz,
    int proj_pad, int pad_before)
{
    int b   = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;   // row = a * chunk_nz + lz
    if (b >= bins || row >= na * chunk_nz) return;
    int a  = row / chunk_nz;
    int lz = row % chunk_nz;
    pad_out[(size_t)row * proj_pad + b + pad_before] =
        all_sinos[((size_t)a * nz + chunk_z0 + lz) * bins + b];
}

__global__ void hbFilterKernel(cufftComplex* __restrict__ spectrum,
                               const float* __restrict__ filter,
                               int complex_size, int batch) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    if (k >= complex_size || b >= batch) return;
    float f = filter[k];
    int idx = b * complex_size + k;
    spectrum[idx].x *= f;
    spectrum[idx].y *= f;
}

__global__ void hbScaleKernel(float* __restrict__ data, int n, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] *= scale;
}

// ---------- 3D backprojection with 3D TEXTURE [P3] + optional AIR-SKIP [P2] ----------
//
// [P3] Текстура — 3D cudaArray с extent (sq, chunk_nz, na). Адресация:
//      tex3D(u, v_row, a). Соседние углы → соседние z-слои текстуры,
//      CUDA-cache для 3D-текстур использует блочно-тайловый layout с хорошей
//      3D-локальностью. На плотных данных это даёт лучший hit-rate, чем
//      исходный 2D layout (sq, na*chunk_nz), где переход a→a+1 прыгал на
//      chunk_nz=64 строк = ~64KB stride.
//
// [P2] Шаблонный параметр USE_AIR_SKIP позволяет компилятору вычеркнуть весь
//      lookup-код, когда air-skip отключён (хост выбирает вариант на основе
//      реально измеренной air_fraction). Для плотных данных без воздуха
//      получаем kernel без лишних global-load'ов.
template <bool USE_AIR_SKIP>
__global__ void hbVolumeBackprojTex3DKernel(
    cudaTextureObject_t  tex_projs,
    const int* __restrict__ u_min_g,       // unused если USE_AIR_SKIP=false
    const int* __restrict__ u_max_g,
    int depth_global,
    int chunk_z0,
    float* __restrict__ vol_out,
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

    // Circular FOV mask
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

    const int z_global = chunk_z0 + v_row;
    const float v_coord = (float)v_row + 0.5f;

    float sum = 0.0f;
    #pragma unroll 4
    for (int a = 0; a < num_angles; ++a) {
        float u = fx * hb_cos_table[a] - fy * hb_sin_table[a] + det_cx;
        int u0 = (int)floorf(u);
        if (u0 < 0 || u0 >= nw - 1) continue;

        if (USE_AIR_SKIP) {
            int bnd_idx = a * depth_global + z_global;
            int umn = u_min_g[bnd_idx];
            int umx = u_max_g[bnd_idx];
            if (u0 < umn || u0 > umx) continue;
        }

        // 3D tex: (u, v_row, a). Соседние углы — соседние слои → лучший cache.
        const float a_coord = (float)a + 0.5f;
        float v0 = tex3D<float>(tex_projs, (float)u0     + 0.5f, v_coord, a_coord);
        float v1 = tex3D<float>(tex_projs, (float)u0 + 1 + 0.5f, v_coord, a_coord);
        float fu = u - (float)u0;
        sum += v0 + (v1 - v0) * fu;
    }

    vol_out[(z * ny + y) * nx + x] = sum;
}

// ---------- Air-Skip boundaries: CPU OpenMP step ----------
//
// Вход — отфильтрованная синограмма ЧАНКА в pinned-host буфере:
//   filtered[(a*chunk_nz + lz) * proj_pad + u]
// Кладёт результат в out_u_min/out_u_max в layout
//   [a*depth_global + (z0+lz)]  (под глобальную индексацию, чтобы kernel
//                                читал по полному [a*depth+z]).
//
// Порог 1% от per-row max-abs (упрощённый аналог K-means k=2 из статьи).
// Margin под side-lobes ramp-фильтра. После filter-domain'а энергия уже
// фактически распределена, поэтому margin может быть умеренным.
static constexpr int AIR_SKIP_MARGIN = 16;

// [P2] Возвращает air_fraction ∈ [0, 1] — долю «воздушных» бинов в чанке.
// Хост использует это значение, чтобы решить, имеет ли смысл launch'ить
// kernel с air-skip lookup'ами или сразу no-lookup вариант (плотные данные).
static float computeAirBoundariesFilteredCPU(
    const float* filtered_chunk,   // [a * chunk_nz + lz][proj_pad]
    int num_angles, int chunk_nz, int proj_pad,
    int sq,                        // ширина активной (square) части в проекции
    int depth_global, int z0,
    int* out_u_min,                // [num_angles * depth_global], записываются строки z0..z0+chunk_nz-1
    int* out_u_max)
{
    const int total = num_angles * chunk_nz;
    long long total_air = 0;
    const long long total_bins = (long long)total * sq;

    #pragma omp parallel for schedule(static) reduction(+:total_air)
    for (int idx = 0; idx < total; ++idx) {
        const int a  = idx / chunk_nz;
        const int lz = idx % chunk_nz;
        const float* row = filtered_chunk + (size_t)idx * proj_pad;

        float row_max = 0.0f;
        for (int u = 0; u < sq; ++u) {
            float v = fabsf(row[u]);
            if (v > row_max) row_max = v;
        }
        const float thr = row_max * 0.01f;

        int umn = sq, umx = -1;
        for (int u = 0; u < sq; ++u) {
            if (fabsf(row[u]) > thr) {
                if (u < umn) umn = u;
                if (u > umx) umx = u;
            }
        }

        const int z_global = z0 + lz;
        const int dst = a * depth_global + z_global;
        if (umx < 0) {
            out_u_min[dst] = sq;
            out_u_max[dst] = -1;
            total_air += sq;
            continue;
        }
        int u_lo = umn - AIR_SKIP_MARGIN;
        int u_hi = umx + AIR_SKIP_MARGIN;
        if (u_lo < 0)      u_lo = 0;
        if (u_hi > sq - 1) u_hi = sq - 1;
        out_u_min[dst] = u_lo;
        out_u_max[dst] = u_hi;
        total_air += (long long)(sq - (u_hi - u_lo + 1));
    }
    return total_bins > 0 ? (float)((double)total_air / (double)total_bins) : 0.0f;
}

// ---------- Point cloud helpers ----------
struct HbVolToPoint {
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
struct HbThreshold {
    float thr;
    __host__ __device__ bool operator()(const ct::Point& p) const { return p.hu > thr; }
};

} // anonymous namespace

// ============================================================
//  Resource management
// ============================================================

HybridBackend::~HybridBackend() { releaseAll(); }

void HybridBackend::releaseAll() const {
    auto free_if = [](float*& p){ if (p) { cudaFree(p); p = nullptr; } };

    free_if(m_d_vol_in); free_if(m_d_vol_out); free_if(m_d_all_sinos);
    for (int s = 0; s < 2; ++s) {
        if (m_d_projs_pad[s]) { cudaFree(m_d_projs_pad[s]); m_d_projs_pad[s] = nullptr; }
        if (m_d_umin[s])      { cudaFree(m_d_umin[s]);      m_d_umin[s]      = nullptr; }
        if (m_d_umax[s])      { cudaFree(m_d_umax[s]);      m_d_umax[s]      = nullptr; }
        if (m_h_filt[s])      { cudaFreeHost(m_h_filt[s]);  m_h_filt[s]      = nullptr; }
        if (m_h_umin[s])      { cudaFreeHost(m_h_umin[s]);  m_h_umin[s]      = nullptr; }
        if (m_h_umax[s])      { cudaFreeHost(m_h_umax[s]);  m_h_umax[s]      = nullptr; }
    }
    free_if(m_d_filter); free_if(m_d_cos); free_if(m_d_sin);
    free_if(m_d_min_hu); free_if(m_d_span); free_if(m_d_slice_minmax);
    free_if(m_d_single_slice); free_if(m_d_single_sino);
    m_singleSliceCap = m_singleSinoCap = 0;
    if (m_d_spectrum) { cudaFree(m_d_spectrum); m_d_spectrum = nullptr; }

    if (m_bpTexObj)  { cudaDestroyTextureObject(m_bpTexObj); m_bpTexObj = 0; }
    if (m_bpArray3D) { cudaFreeArray(m_bpArray3D); m_bpArray3D = nullptr; }

    if (m_planR2C) { cufftDestroy(m_planR2C); m_planR2C = 0; }
    if (m_planC2R) { cufftDestroy(m_planC2R); m_planC2R = 0; }

    if (m_streamsInited) {
        cudaStreamDestroy(m_stream_filt);
        cudaStreamDestroy(m_stream_copy);
        cudaStreamDestroy(m_stream_bp);
        m_stream_filt = m_stream_copy = m_stream_bp = nullptr;
        for (int s = 0; s < 2; ++s) {
            cudaEventDestroy(m_event_filt_done[s]);
            cudaEventDestroy(m_event_d2h_done[s]);
            cudaEventDestroy(m_event_bp_done[s]);
            m_event_filt_done[s] = nullptr;
            m_event_d2h_done [s] = nullptr;
            m_event_bp_done  [s] = nullptr;
        }
        m_streamsInited = false;
    }

    m_volCap = m_sinoCap = m_projPadCap = m_spectrumCap = 0;
    m_filterSize = m_trigAngles = m_depthCap = 0;
    m_planBatch = m_planPadded = 0;
    m_bpArrayW = m_bpArrayH = m_bpArrayD = 0;
    m_currentProjPad = 0;
    m_filtCap = m_bndCap = 0;
}

// [P3] 3D cudaArray под backproj-текстуру с extent (sq, chunk_nz, na).
//   Адресация в kernel: tex3D(u, v_row, a) → adjacent angles = adjacent z-slices.
//   Cache-friendly для 3D-локальности.
void HybridBackend::ensureBackprojTexture3D(size_t sq, size_t chunk_nz, size_t na) const {
    if (m_bpArray3D && m_bpTexObj &&
        m_bpArrayW >= sq && m_bpArrayH >= chunk_nz && m_bpArrayD >= na) return;

    if (m_bpTexObj)  { cudaDestroyTextureObject(m_bpTexObj); m_bpTexObj = 0; }
    if (m_bpArray3D) { cudaFreeArray(m_bpArray3D); m_bpArray3D = nullptr; }

    cudaChannelFormatDesc chDesc = cudaCreateChannelDesc<float>();
    cudaExtent ext = make_cudaExtent(sq, chunk_nz, na);
    GPUCHECK(cudaMalloc3DArray(&m_bpArray3D, &chDesc, ext));

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = m_bpArray3D;
    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0]   = cudaAddressModeClamp;
    texDesc.addressMode[1]   = cudaAddressModeClamp;
    texDesc.addressMode[2]   = cudaAddressModeClamp;
    texDesc.filterMode       = cudaFilterModePoint;
    texDesc.readMode         = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;
    GPUCHECK(cudaCreateTextureObject(&m_bpTexObj, &resDesc, &texDesc, nullptr));

    m_bpArrayW = sq;
    m_bpArrayH = chunk_nz;
    m_bpArrayD = na;
}

void HybridBackend::ensureStreams() const {
    if (m_streamsInited) return;
    GPUCHECK(cudaStreamCreate(&m_stream_filt));
    GPUCHECK(cudaStreamCreate(&m_stream_copy));
    GPUCHECK(cudaStreamCreate(&m_stream_bp));
    for (int s = 0; s < 2; ++s) {
        GPUCHECK(cudaEventCreateWithFlags(&m_event_filt_done[s], cudaEventDisableTiming));
        GPUCHECK(cudaEventCreateWithFlags(&m_event_d2h_done[s],  cudaEventDisableTiming));
        GPUCHECK(cudaEventCreateWithFlags(&m_event_bp_done[s],   cudaEventDisableTiming));
    }
    m_streamsInited = true;
}

void HybridBackend::ensureWorkspace(size_t w, size_t h, size_t d,
                                    size_t num_angles, size_t bins,
                                    size_t chunk_z) const {
    // [MEMORY] m_d_vol_in/out — только под ОДИН чанк, не весь объём.
    //   Раньше: 2 * w*h*d (для 1024³ = 8 ГБ) → не влезало в 6ГБ VRAM,
    //           WDDM начинал свопить device-память через PCIe → backproj
    //           замедлялся в 25× (~130с вместо 5с).
    //   Сейчас: 2 * w*h*chunk_z (для 1024² × 64 = 512 МБ). Этап I стримит
    //           вход чанками с CPU, Этап II пишет выход по чанкам.
    size_t chunk_vol_req = w * h * chunk_z;
    size_t sino_req = num_angles * d * bins;

    const size_t sq       = static_cast<size_t>(std::ceil(std::sqrt(2.0) * (double)bins));
    const size_t proj_pad = std::max<size_t>(64, utils::nextPowerOfTwo(2 * sq));
    const size_t complex_size = proj_pad / 2 + 1;

    const size_t chunk_rows = num_angles * chunk_z;
    const size_t pp_req     = chunk_rows * proj_pad;
    const size_t spec_req   = chunk_rows * complex_size;

    if (m_volCap < chunk_vol_req) {
        if (m_d_vol_in)  { cudaFree(m_d_vol_in);  m_d_vol_in  = nullptr; }
        if (m_d_vol_out) { cudaFree(m_d_vol_out); m_d_vol_out = nullptr; }
        m_volCap = 0;
        GPUCHECK(cudaMalloc(&m_d_vol_in,  chunk_vol_req * sizeof(float)));
        GPUCHECK(cudaMalloc(&m_d_vol_out, chunk_vol_req * sizeof(float)));
        m_volCap = chunk_vol_req;
    }
    if (m_sinoCap < sino_req) {
        if (m_d_all_sinos) { cudaFree(m_d_all_sinos); m_d_all_sinos = nullptr; }
        m_sinoCap = 0;
        GPUCHECK(cudaMalloc(&m_d_all_sinos, sino_req * sizeof(float)));
        m_sinoCap = sino_req;
    }

    if (m_currentProjPad != proj_pad) {
        for (int s = 0; s < 2; ++s) {
            if (m_d_projs_pad[s]) { cudaFree(m_d_projs_pad[s]); m_d_projs_pad[s] = nullptr; }
            if (m_h_filt[s])      { cudaFreeHost(m_h_filt[s]);  m_h_filt[s]      = nullptr; }
        }
        if (m_d_spectrum) { cudaFree(m_d_spectrum); m_d_spectrum = nullptr; }
        m_projPadCap = m_spectrumCap = m_filtCap = 0;
        m_currentProjPad = proj_pad;
    }

    if (m_projPadCap < pp_req) {
        for (int s = 0; s < 2; ++s) {
            if (m_d_projs_pad[s]) { cudaFree(m_d_projs_pad[s]); m_d_projs_pad[s] = nullptr; }
            GPUCHECK(cudaMalloc(&m_d_projs_pad[s], pp_req * sizeof(float)));
        }
        m_projPadCap = pp_req;
    }
    if (m_spectrumCap < spec_req) {
        if (m_d_spectrum) { cudaFree(m_d_spectrum); m_d_spectrum = nullptr; }
        m_spectrumCap = 0;
        GPUCHECK(cudaMalloc(&m_d_spectrum, spec_req * sizeof(cufftComplex)));
        m_spectrumCap = spec_req;
    }

    // [SIMPLIFY] m_h_filt (pinned host буфер для D2H фильтрованного pad'а к CPU
    // для air-boundary анализа) и m_d_umin/m_d_umax/m_h_umin/m_h_umax (boundary
    // tables) больше не нужны — air-skip path отключён. Аллокацию убрали
    // (экономия ~188 МБ pinned RAM на дефолтных параметрах). Сами поля
    // оставлены в .h на случай возвращения air-skip'а в будущем.

    if (m_depthCap < d) {
        if (m_d_min_hu)       { cudaFree(m_d_min_hu);       m_d_min_hu       = nullptr; }
        if (m_d_span)         { cudaFree(m_d_span);         m_d_span         = nullptr; }
        if (m_d_slice_minmax) { cudaFree(m_d_slice_minmax); m_d_slice_minmax = nullptr; }
        m_depthCap = 0;
        GPUCHECK(cudaMalloc(&m_d_min_hu, d * sizeof(float)));
        GPUCHECK(cudaMalloc(&m_d_span,   d * sizeof(float)));
        GPUCHECK(cudaMalloc(&m_d_slice_minmax, 2 * d * sizeof(float)));
        m_depthCap = d;
    }
}

void HybridBackend::ensureFilter(size_t padded_size,
                                  ReconstructionParams::FilterType type) const {
    if (m_filterSize == padded_size && m_filterType == type) return;
    // FilterDesign возвращает спектр размера padded_size/2 + 1 (R2C-форма),
    // НЕ padded_size. Аллоцируем и копируем ровно столько, сколько фактически
    // лежит в векторе — иначе cudaMemcpy читает за концом std::vector.
    std::vector<float> cpu_f = FilterDesign::createFilter(padded_size, type);
    const size_t bytes = cpu_f.size() * sizeof(float);
    if (m_d_filter) cudaFree(m_d_filter);
    GPUCHECK(cudaMalloc(&m_d_filter, bytes));
    GPUCHECK(cudaMemcpy(m_d_filter, cpu_f.data(), bytes, cudaMemcpyHostToDevice));
    m_filterSize = padded_size;
    m_filterType = type;
}

void HybridBackend::ensurePlans(size_t batch, size_t padded_size) const {
    if (m_planBatch == batch && m_planPadded == padded_size) return;
    if (m_planR2C) { cufftDestroy(m_planR2C); m_planR2C = 0; }
    if (m_planC2R) { cufftDestroy(m_planC2R); m_planC2R = 0; }
    m_planBatch = m_planPadded = 0;

    int n[1] = { (int)padded_size };
    CUFFT_CHECK(cufftPlanMany(&m_planR2C, 1, n, NULL, 1, (int)padded_size,
                  NULL, 1, (int)(padded_size/2+1), CUFFT_R2C, (int)batch));
    CUFFT_CHECK(cufftPlanMany(&m_planC2R, 1, n, NULL, 1, (int)(padded_size/2+1),
                  NULL, 1, (int)padded_size, CUFFT_C2R, (int)batch));
    m_planBatch  = batch;
    m_planPadded = padded_size;
}

void HybridBackend::ensureTrigTables(const std::vector<float>& angles_deg) const {
    size_t n = angles_deg.size();
    if (n == 0) return;

    const float first = angles_deg.front();
    const float last  = angles_deg.back();
    if (m_trigAngles == n && m_cachedFirstAngle == first && m_cachedLastAngle == last) return;

    std::vector<float> hcos(n), hsin(n);
    for (size_t a = 0; a < n; ++a) {
        float th = angles_deg[a] * (float)(M_PI / 180.0);
        hcos[a] = std::cos(th);
        hsin[a] = std::sin(th);
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
    cudaMemcpyToSymbol(hb_cos_table, hcos.data(), cnt * sizeof(float));
    cudaMemcpyToSymbol(hb_sin_table, hsin.data(), cnt * sizeof(float));
    m_cachedFirstAngle = first;
    m_cachedLastAngle  = last;
}

bool HybridBackend::isAvailable() const {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

// ============================================================
//  single-slice paths → delegate to OpenMP CPU
//  (PCIe-передача одиночного среза нерентабельна)
// ============================================================

Sinogram HybridBackend::computeSinogram(const Buffer2D& slice, size_t num_angles,
                                        size_t detector_bins, bool /*use_parallel*/) {
    // Single-slice path реализован на GPU. Раньше делегировался на CPU OpenMP —
    // это нормально для одного вызова, но controller вызывает computeSinogram
    // ~depth раз подряд для генерации UI-синограмм (см. UI-loop в
    // CtReconstructionController). На 512³ это ~15 сек CPU-Радона, что
    // ошибочно списывалось на «время реконструкции».
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    Sinogram sino;
    if (slice.empty() || num_angles == 0 || detector_bins == 0) return sino;

    const int w  = (int)slice.width;
    const int h  = (int)slice.height;
    const int na = (int)num_angles;
    const int db = (int)detector_bins;
    const size_t pixels = (size_t)w * h;
    const size_t sino_n = (size_t)na * db;

    sino.data.assign(detector_bins, num_angles, 0.0f);
    sino.angles_deg.resize(num_angles);
    const float step = 180.0f / (float)num_angles;
    for (size_t a = 0; a < num_angles; ++a) sino.angles_deg[a] = step * (float)a;
    sino.detector_spacing_mm = 1.0f;

    // Lazy alloc single-slice scratch
    if (m_singleSliceCap < pixels) {
        if (m_d_single_slice) cudaFree(m_d_single_slice);
        GPUCHECK(cudaMalloc(&m_d_single_slice, pixels * sizeof(float)));
        m_singleSliceCap = pixels;
    }
    if (m_singleSinoCap < sino_n) {
        if (m_d_single_sino) cudaFree(m_d_single_sino);
        GPUCHECK(cudaMalloc(&m_d_single_sino, sino_n * sizeof(float)));
        m_singleSinoCap = sino_n;
    }

    ensureTrigTables(sino.angles_deg);

    GPUCHECK(cudaMemcpy(m_d_single_slice, slice.data.data(),
                        pixels * sizeof(float), cudaMemcpyHostToDevice));

    thrust::device_ptr<float> dp(m_d_single_slice);
    float vmin = *thrust::min_element(dp, dp + pixels);
    float vmax = *thrust::max_element(dp, dp + pixels);
    sino.original_min_hu = vmin;
    sino.original_max_hu = vmax;

    float span = vmax - vmin;
    if (span < 1e-6f) span = 1.0f;
    hbNormalizeKernel<<<((int)pixels + 255) / 256, 256>>>(
        m_d_single_slice, (int)pixels, vmin, 1.0f / span);

    GPUCHECK(cudaMemset(m_d_single_sino, 0, sino_n * sizeof(float)));

    const float cx         = (w - 1) * 0.5f;
    const float cy         = (h - 1) * 0.5f;
    const float det_center = (float)(db / 2);

    dim3 blk(16, 16);
    dim3 grd((w + 15) / 16, (h + 15) / 16);
    // Scatter-Радон: depth=1, z_idx=0. sino rows have layout [a * depth + z][bins]
    // = [a * 1 + 0][bins] = [a][bins] — это ровно наш sino.data layout.
    hbRadonScatterKernel<<<grd, blk>>>(
        m_d_single_slice, m_d_single_sino,
        w, h, /*z_idx=*/0, na, db, /*depth=*/1,
        cx, cy, det_center);

    GPUCHECK(cudaMemcpy(sino.data.data.data(), m_d_single_sino,
                        sino_n * sizeof(float), cudaMemcpyDeviceToHost));
    return sino;
}

Buffer2D HybridBackend::reconstructSlice(const Sinogram& sinogram, size_t output_size,
                                         const ReconstructionParams& params) {
    return FilteredBackprojection::reconstruct(sinogram, output_size, params);
}

// ============================================================
//  reconstructVolume — ультимативный гибридный CPU+GPU pipeline
// ============================================================

void HybridBackend::reconstructVolume(const Volume& input_volume,
                                      Volume& out_reconstruction,
                                      const ReconstructionParams& params,
                                      std::function<void(int, const Buffer2D&)> onSliceDone) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    if (input_volume.empty()) return;

    const int nx   = (int)input_volume.width;
    const int ny   = (int)input_volume.height;
    const int nz   = (int)input_volume.depth;
    const int na   = (int)params.num_angles;
    const int bins = nx;

    out_reconstruction.assign(nx, ny, nz, 0.0f);
    out_reconstruction.x_coords = input_volume.x_coords;
    out_reconstruction.y_coords = input_volume.y_coords;
    out_reconstruction.z_coords = input_volume.z_coords;

    const size_t sq           = static_cast<size_t>(std::ceil(std::sqrt(2.0) * (double)bins));
    const size_t pad_before   = (sq / 2) - ((size_t)bins / 2);
    const size_t proj_pad     = std::max<size_t>(64, utils::nextPowerOfTwo(2 * sq));
    const size_t complex_size = proj_pad / 2 + 1;

    // Углы
    std::vector<float> angles(na);
    const float step = 180.0f / (float)na;
    for (int i = 0; i < na; ++i) angles[i] = step * (float)i;

    // CHUNK_Z = 64 (стабильно, подтверждено на 256³/512³).
    constexpr int CHUNK_Z = 64;

    ensureWorkspace((size_t)nx, (size_t)ny, (size_t)nz,
                    (size_t)na, (size_t)bins, (size_t)CHUNK_Z);
    ensureFilter(proj_pad, params.filter);
    ensureTrigTables(angles);
    ensureStreams();

    const float cx      = (nx - 1) * 0.5f;
    const float cy      = (ny - 1) * 0.5f;
    const float cz      = (nz - 1) * 0.5f;
    const float det_cx  = (float)(sq / 2);
    const float det_ctr = (float)(bins / 2);

    // ====================================================================
    // ЭТАП I — Радон-проекция объёма на GPU. Streaming + async.
    //
    // [MEMORY] Вход стримим CPU→GPU чанками по CHUNK_Z слайсов. m_d_vol_in
    //   держит только один чанк (256 МБ для 1024², а не 4 ГБ за весь объём).
    //   Это критично на consumer GPU с ≤6 ГБ VRAM, иначе WDDM начинает
    //   свопить device-память → backproj в 25× медленнее.
    //
    // На каждый слайс:
    //   1. thrust::minmax_element(par.on(stream)) — async, результат на device
    //   2. DtoD copy (vmin, vmax) → m_d_slice_minmax[2*z..]
    //   3. hbNormalizeFromDevicePairKernel — нормализация без host-sync
    //   4. hbRadonGatherKernel — поток=(bin,a), без атомиков, coalesced writes
    //      → пишет в правильный глобальный z в m_d_all_sinos
    // После всех чанков — hbMinMaxToMinSpanKernel конвертирует (vmin,vmax)[nz]
    // → (min[nz], span[nz]) для Этапа II.
    // ====================================================================
    auto t_sino_start = std::chrono::steady_clock::now();

    const int total_rows = na * nz;
    GPUCHECK(cudaMemsetAsync(m_d_all_sinos, 0,
                              (size_t)total_rows * bins * sizeof(float),
                              m_stream_filt));

    dim3 blk_r(16, 16);
    dim3 grd_r((bins + 15) / 16, (na + 15) / 16);
    const int nxy = nx * ny;
    const size_t slice_bytes = (size_t)nxy * sizeof(float);

    for (int chunk_z0 = 0; chunk_z0 < nz; chunk_z0 += CHUNK_Z) {
        const int chunk_nz = std::min(CHUNK_Z, nz - chunk_z0);

        // H2D очередного чанка слайсов в m_d_vol_in (chunk-sized).
        const float* h_chunk = input_volume.data.data() + (size_t)chunk_z0 * nxy;
        GPUCHECK(cudaMemcpyAsync(m_d_vol_in, h_chunk,
                                  (size_t)chunk_nz * slice_bytes,
                                  cudaMemcpyHostToDevice, m_stream_filt));

        // Обработка каждого слайса чанка.
        for (int lz = 0; lz < chunk_nz; ++lz) {
            const int z_global = chunk_z0 + lz;
            float* d_slice = m_d_vol_in + (size_t)lz * nxy;

            // 1. Async minmax → device memory
            thrust::device_ptr<float> dp(d_slice);
            auto mm = thrust::minmax_element(
                thrust::cuda::par.on(m_stream_filt),
                dp, dp + nxy);
            // 2. DtoD copy в m_d_slice_minmax[2*z_global..2*z_global+1]
            GPUCHECK(cudaMemcpyAsync(m_d_slice_minmax + 2 * z_global,
                                      thrust::raw_pointer_cast(mm.first),
                                      sizeof(float),
                                      cudaMemcpyDeviceToDevice, m_stream_filt));
            GPUCHECK(cudaMemcpyAsync(m_d_slice_minmax + 2 * z_global + 1,
                                      thrust::raw_pointer_cast(mm.second),
                                      sizeof(float),
                                      cudaMemcpyDeviceToDevice, m_stream_filt));

            // 3. Нормализация (читает vmin/vmax с device).
            hbNormalizeFromDevicePairKernel<<<(nxy + 255) / 256, 256, 0, m_stream_filt>>>(
                d_slice, nxy, m_d_slice_minmax + 2 * z_global);

            // 4. Gather-Radon → пишет в m_d_all_sinos на ГЛОБАЛЬНЫЙ z_global.
            hbRadonGatherKernel<<<grd_r, blk_r, 0, m_stream_filt>>>(
                d_slice, m_d_all_sinos,
                nx, ny, na, bins,
                cx, cy, det_ctr,
                /*z_slice=*/z_global, /*depth=*/nz);
        }
    }

    // (vmin,vmax)[nz] → (min[nz], span[nz]) для Этапа II.
    hbMinMaxToMinSpanKernel<<<(nz + 255) / 256, 256, 0, m_stream_filt>>>(
        m_d_slice_minmax, m_d_min_hu, m_d_span, nz);

    GPUCHECK(cudaStreamSynchronize(m_stream_filt));

    auto t_sino_end = std::chrono::steady_clock::now();
    m_lastSinogramTimeMs = std::chrono::duration<double, std::milli>(t_sino_end - t_sino_start).count();

    // ====================================================================
    // ЭТАП II — Pipelined chunk FBP с ping-pong overlap (filter ║ backproj).
    //
    //   На каждой итерации i:
    //     [GPU stream_filt] : filter(i+1) → pad[(i+1)%2]    (FFT/iFFT/scale)
    //     [GPU stream_bp]   : backproj(i) ← pad[i%2] → 3D-tex → vol_out
    //                                                       → D2H output
    //
    //   Реальный GPU overlap: filter(i+1) ║ backproj(i) (разные streams).
    //
    // [SIMPLIFY] CPU air-boundary анализ отключён — для синтетических CT
    // данных он почти всегда давал use_skip=false, при этом блокировал
    // backproj на ~30мс/чанк CPU-работой и тратил PCIe на D2H фильтрованного
    // pad'а. Для медицинских DICOM с большим воздухом можно вернуть
    // (см. git history launch_async_boundaries).
    // ====================================================================
    const int n_chunks = (nz + CHUNK_Z - 1) / CHUNK_Z;
    const float pi_scale = (float)(M_PI / (2.0 * na));

    double recon_accum_ms = 0.0;
    auto t_recon_start = std::chrono::steady_clock::now();

    int chunk_size[2]    = { 0, 0 };
    int chunk_z0_for[2]  = { 0, 0 };

    // ====================================================================
    // Per-chunk profiling: cudaEvent для GPU-фаз + std::chrono для CPU bnd.
    // Печатает раскладку в stderr после полного pipeline'а.
    // ====================================================================
    struct ChunkProfile {
        cudaEvent_t filt_start    = nullptr;   // stream_filt
        cudaEvent_t filt_end      = nullptr;   // stream_filt
        cudaEvent_t d2h_pad_end   = nullptr;   // stream_copy
        cudaEvent_t bp_start      = nullptr;   // stream_bp (перед memcpy3d)
        cudaEvent_t memcpy3d_end  = nullptr;   // stream_bp
        cudaEvent_t bp_kernel_end = nullptr;   // stream_bp
        cudaEvent_t denorm_end    = nullptr;   // stream_bp
        cudaEvent_t d2h_out_end   = nullptr;   // stream_bp
        std::chrono::steady_clock::time_point cpu_wait_start;
        std::chrono::steady_clock::time_point cpu_wait_end;
        float air_frac    = 0.0f;
        bool  use_skip    = false;
    };
    std::vector<ChunkProfile> profiles(n_chunks);
    for (auto& p : profiles) {
        cudaEventCreate(&p.filt_start);
        cudaEventCreate(&p.filt_end);
        cudaEventCreate(&p.d2h_pad_end);
        cudaEventCreate(&p.bp_start);
        cudaEventCreate(&p.memcpy3d_end);
        cudaEventCreate(&p.bp_kernel_end);
        cudaEventCreate(&p.denorm_end);
        cudaEventCreate(&p.d2h_out_end);
    }

    auto launch_filter = [&](int chunk_idx, int slot) {
        const int z0       = chunk_idx * CHUNK_Z;
        const int chunk_nz = std::min(CHUNK_Z, nz - z0);
        const int chunk_rows = na * chunk_nz;
        chunk_size[slot]   = chunk_nz;
        chunk_z0_for[slot] = z0;
        ChunkProfile& prof = profiles[chunk_idx];

        ensurePlans((size_t)chunk_rows, proj_pad);

        // Перед записью pad[slot] дождаться, что предыдущий backproj его дочитал.
        cudaStreamWaitEvent(m_stream_filt, m_event_bp_done[slot], 0);

        cudaEventRecord(prof.filt_start, m_stream_filt);

        // [P1] Объединённый gather+pack одним kernel'ом вместо na отдельных
        //      cudaMemcpy2DAsync.
        GPUCHECK(cudaMemsetAsync(m_d_projs_pad[slot], 0,
                                 (size_t)chunk_rows * proj_pad * sizeof(float),
                                 m_stream_filt));
        {
            dim3 blk(32, 8);
            dim3 grd((bins + 31) / 32, (chunk_rows + 7) / 8);
            hbGatherPackKernel<<<grd, blk, 0, m_stream_filt>>>(
                m_d_all_sinos, m_d_projs_pad[slot],
                na, nz, bins,
                z0, chunk_nz,
                (int)proj_pad, (int)pad_before);
        }

        // FFT → filter → iFFT → scale на stream_filt
        CUFFT_CHECK(cufftSetStream(m_planR2C, m_stream_filt));
        CUFFT_CHECK(cufftSetStream(m_planC2R, m_stream_filt));
        CUFFT_CHECK(cufftExecR2C(m_planR2C, (cufftReal*)m_d_projs_pad[slot], m_d_spectrum));
        {
            dim3 blk(16, 16);
            dim3 grd(((int)complex_size + 15) / 16, (chunk_rows + 15) / 16);
            hbFilterKernel<<<grd, blk, 0, m_stream_filt>>>(
                m_d_spectrum, m_d_filter, (int)complex_size, chunk_rows);
        }
        CUFFT_CHECK(cufftExecC2R(m_planC2R, m_d_spectrum, (cufftReal*)m_d_projs_pad[slot]));
        {
            int n_total = chunk_rows * (int)proj_pad;
            float fft_scale = 1.0f / (float)proj_pad;
            hbScaleKernel<<<(n_total + 255) / 256, 256, 0, m_stream_filt>>>(
                m_d_projs_pad[slot], n_total, fft_scale);
        }
        cudaEventRecord(prof.filt_end, m_stream_filt);
        cudaEventRecord(m_event_filt_done[slot], m_stream_filt);

        // [SIMPLIFY] D2H фильтрованного pad'а к хосту больше не делаем — он был
        // нужен только для CPU air-boundary анализа, который теперь отключён.
        // Сохраняем event_d2h_done для совместимости с unused профайлингом.
        cudaEventRecord(prof.d2h_pad_end, m_stream_filt);
        cudaEventRecord(m_event_d2h_done[slot], m_stream_filt);
    };

    auto launch_backproj = [&](int chunk_idx, int slot) {
        const int chunk_nz   = chunk_size[slot];
        const int z0         = chunk_z0_for[slot];
        ChunkProfile& prof   = profiles[chunk_idx];

        // [SIMPLIFY] CPU air-boundary path отключён — для синтетических CT-данных
        // air_frac обычно < AIR_SKIP_THRESHOLD, kernel всегда выбирал no-skip вариант,
        // а CPU-работа блокировала backproj (~30мс на чанк ждали впустую). Если
        // в будущем нужно для медицинских DICOM с большим воздухом — раскомментировать
        // launch_async_boundaries и условную ветку ниже.
        prof.cpu_wait_start = prof.cpu_wait_end = std::chrono::steady_clock::now();
        prof.air_frac = 0.0f;
        prof.use_skip = false;

        // stream_bp ждёт, что filter в pad[slot] завершился
        cudaStreamWaitEvent(m_stream_bp, m_event_filt_done[slot], 0);
        cudaEventRecord(prof.bp_start, m_stream_bp);

        // [P3] Заливаем pad[slot] в 3D-текстуру через cudaMemcpy3DAsync.
        ensureBackprojTexture3D((size_t)sq, (size_t)chunk_nz, (size_t)na);
        {
            cudaMemcpy3DParms p = {};
            p.srcPtr = make_cudaPitchedPtr(
                m_d_projs_pad[slot],
                proj_pad * sizeof(float),
                sq       * sizeof(float),
                (size_t)chunk_nz);
            p.dstArray = m_bpArray3D;
            p.extent   = make_cudaExtent((size_t)sq, (size_t)chunk_nz, (size_t)na);
            p.kind     = cudaMemcpyDeviceToDevice;
            GPUCHECK(cudaMemcpy3DAsync(&p, m_stream_bp));
        }
        cudaEventRecord(prof.memcpy3d_end, m_stream_bp);

        const size_t chunk_vol_bytes = (size_t)nx * ny * chunk_nz * sizeof(float);
        GPUCHECK(cudaMemsetAsync(m_d_vol_out, 0, chunk_vol_bytes, m_stream_bp));

        const float chunk_cz     = (chunk_nz - 1) * 0.5f;
        const float chunk_det_cy = chunk_cz;
        dim3 blk(8, 8, 4);
        dim3 grd((nx + blk.x - 1) / blk.x,
                 (ny + blk.y - 1) / blk.y,
                 (chunk_nz + blk.z - 1) / blk.z);
        hbVolumeBackprojTex3DKernel<false><<<grd, blk, 0, m_stream_bp>>>(
            m_bpTexObj,
            nullptr, nullptr,
            nz, z0,
            m_d_vol_out,
            nx, ny, chunk_nz, (int)sq, na,
            cx, cy, chunk_cz, det_cx, chunk_det_cy);
        cudaEventRecord(prof.bp_kernel_end, m_stream_bp);

        // [SIMPLIFY] Denormalize: m_d_min_hu/m_d_span уже заполнены на device
        // в Этапе I через hbMinMaxToMinSpanKernel — H2D больше не нужен.
        int n_chunk_vox = nx * ny * chunk_nz;
        hbDenormalizeVolumeKernel<<<(n_chunk_vox + 255) / 256, 256, 0, m_stream_bp>>>(
            m_d_vol_out, m_d_min_hu + z0, m_d_span + z0, nx * ny, chunk_nz, pi_scale);
        cudaEventRecord(prof.denorm_end, m_stream_bp);

        // D2H output chunk на stream_bp
        float* dst_ptr = out_reconstruction.data.data() + (size_t)z0 * nx * ny;
        GPUCHECK(cudaMemcpyAsync(dst_ptr, m_d_vol_out, chunk_vol_bytes,
                                 cudaMemcpyDeviceToHost, m_stream_bp));
        cudaEventRecord(prof.d2h_out_end, m_stream_bp);

        cudaEventRecord(m_event_bp_done[slot], m_stream_bp);
    };

    // -------------------- pipeline run --------------------
    // Prologue: запустить filter для чанка 0
    launch_filter(0, 0);

    for (int i = 0; i < n_chunks; ++i) {
        const int slot      = i % 2;
        const int slot_next = (i + 1) % 2;

        // Запустить filter для следующего чанка
        // (бежит параллельно с backproj текущего).
        if (i + 1 < n_chunks) {
            launch_filter(i + 1, slot_next);
        }

        // Backproj текущего чанка (ждёт filter[slot]).
        launch_backproj(i, slot);

        // Дождёмся D2H output текущего чанка перед onSliceDone.
        GPUCHECK(cudaStreamSynchronize(m_stream_bp));

        if (onSliceDone) {
            const int z0 = chunk_z0_for[slot];
            for (int lz = 0; lz < chunk_size[slot]; ++lz)
                onSliceDone(z0 + lz, out_reconstruction.getSlice((size_t)(z0 + lz)));
        }
    }

    GPUCHECK(cudaDeviceSynchronize());

    auto t_recon_end = std::chrono::steady_clock::now();
    recon_accum_ms = std::chrono::duration<double, std::milli>(t_recon_end - t_recon_start).count();
    m_lastReconstructionTimeMs = recon_accum_ms;

    // ====================================================================
    // Печать per-phase профиля. Для каждой фазы суммируем по всем чанкам и
    // печатаем раскладку первого чанка. Это даёт картину, где время теряется.
    // ====================================================================
    {
        double sum_filt = 0, sum_d2h_pad = 0, sum_cpu_wait = 0;
        double sum_memcpy3d = 0, sum_bp_kernel = 0, sum_denorm = 0, sum_d2h_out = 0;
        std::fprintf(stderr, "\n=== HybridBackend per-chunk profile (n_chunks=%d) ===\n", n_chunks);
        for (int i = 0; i < n_chunks; ++i) {
            const ChunkProfile& p = profiles[i];
            float ms_filt = 0, ms_d2h_pad_total = 0, ms_memcpy3d = 0, ms_bp_kernel = 0, ms_denorm = 0, ms_d2h_out = 0;
            cudaEventElapsedTime(&ms_filt,       p.filt_start,    p.filt_end);
            cudaEventElapsedTime(&ms_d2h_pad_total, p.filt_start, p.d2h_pad_end);  // filter+D2H from filt_start
            cudaEventElapsedTime(&ms_memcpy3d,   p.bp_start,      p.memcpy3d_end);
            cudaEventElapsedTime(&ms_bp_kernel,  p.memcpy3d_end,  p.bp_kernel_end);
            cudaEventElapsedTime(&ms_denorm,    p.bp_kernel_end,  p.denorm_end);
            cudaEventElapsedTime(&ms_d2h_out,   p.denorm_end,     p.d2h_out_end);
            float ms_d2h_pad = ms_d2h_pad_total - ms_filt;
            double ms_cpu_wait = std::chrono::duration<double, std::milli>(p.cpu_wait_end - p.cpu_wait_start).count();

            sum_filt       += ms_filt;
            sum_d2h_pad    += ms_d2h_pad;
            sum_cpu_wait   += ms_cpu_wait;
            sum_memcpy3d   += ms_memcpy3d;
            sum_bp_kernel  += ms_bp_kernel;
            sum_denorm     += ms_denorm;
            sum_d2h_out    += ms_d2h_out;

            if (i == 0) {
                const int z0_i0 = 0 * CHUNK_Z;
                const int nz_i0 = std::min((int)CHUNK_Z, nz - z0_i0);
                std::fprintf(stderr,
                    "chunk[0] z0=%d nz=%d air=%.3f use_skip=%d:\n"
                    "  filter(FFT/iFFT)  : %7.2f ms\n"
                    "  D2H pad (post-fil): %7.2f ms\n"
                    "  CPU bnd wait      : %7.2f ms  (host-side block before backproj)\n"
                    "  Memcpy3D pad->arr : %7.2f ms\n"
                    "  backproj kernel   : %7.2f ms\n"
                    "  denorm            : %7.2f ms\n"
                    "  D2H output chunk  : %7.2f ms\n",
                    z0_i0, nz_i0, p.air_frac, (int)p.use_skip,
                    ms_filt, ms_d2h_pad, ms_cpu_wait,
                    ms_memcpy3d, ms_bp_kernel, ms_denorm, ms_d2h_out);
            }
        }
        std::fprintf(stderr,
            "TOTAL (sum of %d chunks):\n"
            "  filter        : %7.1f ms\n"
            "  D2H pad       : %7.1f ms\n"
            "  CPU bnd wait  : %7.1f ms\n"
            "  Memcpy3D      : %7.1f ms\n"
            "  backproj      : %7.1f ms\n"
            "  denorm        : %7.1f ms\n"
            "  D2H output    : %7.1f ms\n"
            "  recon total   : %7.1f ms (m_lastReconstructionTimeMs)\n",
            n_chunks,
            sum_filt, sum_d2h_pad, sum_cpu_wait,
            sum_memcpy3d, sum_bp_kernel, sum_denorm, sum_d2h_out,
            recon_accum_ms);
        std::fflush(stderr);
    }
    for (auto& p : profiles) {
        cudaEventDestroy(p.filt_start);
        cudaEventDestroy(p.filt_end);
        cudaEventDestroy(p.d2h_pad_end);
        cudaEventDestroy(p.bp_start);
        cudaEventDestroy(p.memcpy3d_end);
        cudaEventDestroy(p.bp_kernel_end);
        cudaEventDestroy(p.denorm_end);
        cudaEventDestroy(p.d2h_out_end);
    }
}

// ============================================================
//  extractPointCloud — Thrust (как в CUDABackend)
// ============================================================

PointCloud HybridBackend::extractPointCloud(const Volume& vol, float threshold) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    PointCloud cloud;
    if (vol.empty()) return cloud;

    const size_t N = vol.data.size();
    thrust::device_vector<float> d_vol  = vol.data;
    thrust::device_vector<float> d_xcrd = vol.x_coords;
    thrust::device_vector<float> d_ycrd = vol.y_coords;
    thrust::device_vector<float> d_zcrd = vol.z_coords;

    HbVolToPoint op;
    op.vol_data = thrust::raw_pointer_cast(d_vol.data());
    op.w = (int)vol.width; op.h = (int)vol.height;
    op.xc = vol.x_coords.empty() ? nullptr : thrust::raw_pointer_cast(d_xcrd.data());
    op.yc = vol.y_coords.empty() ? nullptr : thrust::raw_pointer_cast(d_ycrd.data());
    op.zc = vol.z_coords.empty() ? nullptr : thrust::raw_pointer_cast(d_zcrd.data());

    HbThreshold pred; pred.thr = threshold;

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
