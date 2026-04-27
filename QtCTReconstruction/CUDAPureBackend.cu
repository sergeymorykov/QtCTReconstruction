#include "CUDAPureBackend.h"
#include "FilterDesign.h"
#include "Utils.h"
#include "RadonTransform.h"
#include "FilteredBackprojection.h"

#include <cmath>
#include <iostream>
#include <vector>
#include <chrono>

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

#define GPUCHECK(call) \
    do { \
        cudaError_t _e = (call); \
        if (_e != cudaSuccess) \
            std::cerr << "CUDA err " << cudaGetErrorString(_e) \
                      << " @ " << __FILE__ << ":" << __LINE__ << "\n"; \
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
    if (m_d_spectrum) { cudaFree(m_d_spectrum); m_d_spectrum = nullptr; }
    if (m_planR2C) { cufftDestroy(m_planR2C); m_planR2C = 0; }
    if (m_planC2R) { cufftDestroy(m_planC2R); m_planC2R = 0; }
    m_volCap = m_sinoCap = m_projPadCap = m_spectrumCap = 0;
    m_filterSize = m_trigAngles = m_depthCap = 0;
    m_planBatch = m_planPadded = 0;
}

void CUDAPureBackend::ensureWorkspace(size_t w, size_t h, size_t d,
                                       size_t num_angles, size_t bins) const {
    size_t vol_req  = w * h * d;
    size_t sino_req = num_angles * d * bins;

    const size_t sq = static_cast<size_t>(std::ceil(std::sqrt(2.0) * (double)bins));
    const size_t ppf = 2;
    const size_t proj_pad = std::max<size_t>(64, utils::nextPowerOfTwo(ppf * sq));
    size_t proj_pad_req = num_angles * d * proj_pad;
    size_t spec_req     = num_angles * d * (proj_pad / 2 + 1);

    // vol_in and vol_out share the same required size; allocate together
    if (m_volCap < vol_req) {
        if (m_d_vol_in)  cudaFree(m_d_vol_in);
        if (m_d_vol_out) cudaFree(m_d_vol_out);
        GPUCHECK(cudaMalloc(&m_d_vol_in,  vol_req * sizeof(float)));
        GPUCHECK(cudaMalloc(&m_d_vol_out, vol_req * sizeof(float)));
        m_volCap = vol_req;
    }

    if (m_sinoCap < sino_req) {
        if (m_d_all_sinos) cudaFree(m_d_all_sinos);
        GPUCHECK(cudaMalloc(&m_d_all_sinos, sino_req * sizeof(float)));
        m_sinoCap = sino_req;
    }

    if (m_projPadCap < proj_pad_req) {
        if (m_d_projs_pad) cudaFree(m_d_projs_pad);
        GPUCHECK(cudaMalloc(&m_d_projs_pad, proj_pad_req * sizeof(float)));
        m_projPadCap = proj_pad_req;
    }

    if (m_spectrumCap < spec_req) {
        if (m_d_spectrum) cudaFree(m_d_spectrum);
        GPUCHECK(cudaMalloc(&m_d_spectrum, spec_req * sizeof(cufftComplex)));
        m_spectrumCap = spec_req;
    }

    if (m_depthCap < d) {
        if (m_d_min_hu) cudaFree(m_d_min_hu);
        if (m_d_span)   cudaFree(m_d_span);
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
    if (m_planR2C) cufftDestroy(m_planR2C);
    if (m_planC2R) cufftDestroy(m_planC2R);
    int n[1] = { (int)padded_size };
    cufftPlanMany(&m_planR2C, 1, n, NULL, 1, (int)padded_size,
                  NULL, 1, (int)(padded_size/2+1), CUFFT_R2C, (int)batch);
    cufftPlanMany(&m_planC2R, 1, n, NULL, 1, (int)(padded_size/2+1),
                  NULL, 1, (int)padded_size, CUFFT_C2R, (int)batch);
    m_planBatch  = batch;
    m_planPadded = padded_size;
}

void CUDAPureBackend::ensureTrigTables(const std::vector<float>& angles_deg) const {
    if (m_cachedAnglesDeg == angles_deg) return;
    size_t n = angles_deg.size();
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
    m_cachedAnglesDeg = angles_deg;
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

    // Allocate / reuse GPU workspace
    ensureWorkspace((size_t)nx, (size_t)ny, (size_t)nz, (size_t)na, (size_t)bins);
    ensureFilter(proj_pad, params.filter);
    ensurePlans((size_t)total_rows, proj_pad);
    ensureTrigTables(angles);

    const float cx      = (nx - 1) * 0.5f;
    const float cy      = (ny - 1) * 0.5f;
    const float cz      = (nz - 1) * 0.5f;
    // det_cx matches FilteredBackprojection.cpp:89 — integer division sq/2
    const float det_cx  = (float)(sq / 2);
    const float det_cy  = cz;                // = (nz-1)*0.5 (matches ProjectionGeometry fix)
    // Radon scatter bin center uses same integer-division convention
    const float det_ctr = (float)(bins / 2); // for Radon scatter bin centering

    // --------------------------------------------------------
    // Phase 1: Upload full volume H→D (single transfer)
    // --------------------------------------------------------
    auto t0 = std::chrono::steady_clock::now();

    const size_t vol_bytes = (size_t)nx * ny * nz * sizeof(float);
    GPUCHECK(cudaMemcpy(m_d_vol_in, input_volume.data.data(), vol_bytes, cudaMemcpyHostToDevice));

    // --------------------------------------------------------
    // Phase 2: Per-slice normalize + scatter Radon
    //          All sinogram rows stored in m_d_all_sinos:
    //          row = a * nz + z_idx,  width = bins
    // --------------------------------------------------------
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

    auto t1 = std::chrono::steady_clock::now();
    m_lastSinogramTimeMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // --------------------------------------------------------
    // Phase 3: Pack sinograms into zero-padded projection buffer
    //          m_d_projs_pad: row = a*nz+z, width = proj_pad
    // --------------------------------------------------------
    GPUCHECK(cudaMemset(m_d_projs_pad, 0, (size_t)total_rows * proj_pad * sizeof(float)));
    {
        dim3 blk(16, 16);
        dim3 grd((total_rows + 15) / 16, (bins + 15) / 16);
        gpPackKernel<<<grd, blk>>>(
            m_d_all_sinos, m_d_projs_pad,
            total_rows, bins, (int)proj_pad, (int)pad_before);
    }

    // --------------------------------------------------------
    // Phase 4: Batch FFT over all (na × nz) rows simultaneously
    // --------------------------------------------------------
    cufftExecR2C(m_planR2C, (cufftReal*)m_d_projs_pad, m_d_spectrum);

    {
        dim3 blk(16, 16);
        dim3 grd(((int)complex_size + 15) / 16, (total_rows + 15) / 16);
        gpFilterKernel<<<grd, blk>>>(m_d_spectrum, m_d_filter, (int)complex_size, total_rows);
    }

    cufftExecC2R(m_planC2R, m_d_spectrum, (cufftReal*)m_d_projs_pad);

    // Scale after iFFT (cuFFT C2R doesn't normalize)
    {
        int n_total = total_rows * (int)proj_pad;
        float fft_scale = 1.0f / (float)proj_pad;
        gpScaleKernel<<<(n_total + 255) / 256, 256>>>(m_d_projs_pad, n_total, fft_scale);
    }

    // --------------------------------------------------------
    // Phase 5: 3D volumetric backprojection
    //          Reads m_d_projs_pad, writes m_d_vol_out
    // --------------------------------------------------------
    {
        dim3 blk(8, 8, 4);
        dim3 grd((nx + blk.x - 1) / blk.x,
                 (ny + blk.y - 1) / blk.y,
                 (nz + blk.z - 1) / blk.z);
        GPUCHECK(cudaMemset(m_d_vol_out, 0, vol_bytes));
        gpVolumeBackprojectionKernel<<<grd, blk>>>(
            m_d_vol_out, m_d_projs_pad,
            nx, ny, nz, (int)sq, (int)proj_pad, na,
            cx, cy, cz, det_cx, det_cy);
        GPUCHECK(cudaDeviceSynchronize());
    }

    // --------------------------------------------------------
    // Phase 6: Per-slice scale (π/2N) and denormalize on GPU
    // --------------------------------------------------------
    GPUCHECK(cudaMemcpy(m_d_min_hu, h_min.data(),  nz * sizeof(float), cudaMemcpyHostToDevice));
    GPUCHECK(cudaMemcpy(m_d_span,   h_span.data(), nz * sizeof(float), cudaMemcpyHostToDevice));

    const float pi_scale = (float)(M_PI / (2.0 * na));
    int n_total_vox = nx * ny * nz;
    gpDenormalizeVolumeKernel<<<(n_total_vox + 255) / 256, 256>>>(
        m_d_vol_out, m_d_min_hu, m_d_span, nx * ny, nz, pi_scale);

    // --------------------------------------------------------
    // Phase 7: Download reconstructed volume D→H (single transfer)
    // --------------------------------------------------------
    GPUCHECK(cudaMemcpy(out_reconstruction.data.data(), m_d_vol_out, vol_bytes, cudaMemcpyDeviceToHost));

    // Optional slice callbacks
    if (onSliceDone) {
        for (int z = 0; z < nz; ++z)
            onSliceDone(z, out_reconstruction.getSlice(z));
    }
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
