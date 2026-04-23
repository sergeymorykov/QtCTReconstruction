#include "OptimizedBackprojectionCPU.h"
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <immintrin.h>

namespace ct {

namespace {

// Backprojects one row of volume voxels using AVX2 gather + FMA.
// Processes 8 voxels per iteration; scalar fallback for remainder.
inline void backproject_row_avx2(float* vol_row, const float* proj_row,
                                  float u0_base, float du, int x_start, int x_end) {
    int x = x_start;
    float u_base = static_cast<float>(x_start) * du + u0_base;

    __m256 v_du  = _mm256_set1_ps(du);
    __m256 v_u   = _mm256_add_ps(_mm256_set1_ps(u_base),
                    _mm256_mul_ps(v_du, _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f)));
    __m256 v_du8 = _mm256_set1_ps(du * 8.0f);

    for (; x <= x_end - 8; x += 8) {
        __m256i v_u0   = _mm256_cvttps_epi32(v_u);
        __m256  v_fu   = _mm256_sub_ps(v_u, _mm256_cvtepi32_ps(v_u0));
        __m256  v_p0   = _mm256_i32gather_ps(proj_row, v_u0, 4);
        __m256  v_p1   = _mm256_i32gather_ps(proj_row, _mm256_add_epi32(v_u0, _mm256_set1_epi32(1)), 4);
        __m256  v_interp = _mm256_fmadd_ps(v_fu, _mm256_sub_ps(v_p1, v_p0), v_p0);
        __m256  v_vol  = _mm256_loadu_ps(&vol_row[x]);
        _mm256_storeu_ps(&vol_row[x], _mm256_add_ps(v_vol, v_interp));
        v_u = _mm256_add_ps(v_u, v_du8);
    }

    // Scalar remainder
    float u = static_cast<float>(x) * du + u0_base;
    for (; x < x_end; ++x) {
        const int u0 = static_cast<int>(u);
        const float fu = u - static_cast<float>(u0);
        vol_row[x] += proj_row[u0] + fu * (proj_row[u0 + 1] - proj_row[u0]);
        u += du;
    }
}

} // namespace

void OptimizedBackprojectionCPU::reconstruct(Volume& volume,
                                            const Buffer2D& projections,
                                            const std::vector<ProjectionMatrix>& matrices,
                                            const CTGeometry& geom,
                                            int /*batch_size_ignored*/) {
    if (volume.empty() || projections.empty() || matrices.empty()) return;

    int nx = geom.nx;
    int ny = geom.ny;
    int nz = geom.nz;
    int nw = geom.nw;
    int nh = geom.nh;
    int total_projections = static_cast<int>(matrices.size());

    std::fill(volume.data.begin(), volume.data.end(), 0.0f);

    struct ProjParams {
        float u_base_z0;
        float du_dx, du_dy, du_dz;
        float v_base, dv_dz;
    };
    std::vector<ProjParams> all_params(total_projections);
    for (int p = 0; p < total_projections; ++p) {
        const auto& m = matrices[p].data;
        all_params[p].du_dx    = m[0][0];
        all_params[p].du_dy    = m[0][1];
        all_params[p].du_dz    = m[0][2];
        all_params[p].u_base_z0 = m[0][3];
        all_params[p].dv_dz    = m[1][2];
        all_params[p].v_base   = m[1][3];
    }

    const int nz_half = (nz + 1) / 2;
    const float* projs_raw = projections.data.data();
    const size_t slice_size = static_cast<size_t>(nx) * ny;

    // z-slices have equal work → static schedule is faster than dynamic
    #pragma omp parallel for schedule(static)
    for (int z = 0; z < nz_half; ++z) {
        const int z_sym = nz - 1 - z;
        const bool has_sym = (z != z_sym);
        const float fz     = static_cast<float>(z);
        const float fz_sym = static_cast<float>(z_sym);

        float* vol_slice_p = &volume.data[static_cast<size_t>(z)     * slice_size];
        float* vol_slice_n = has_sym ? &volume.data[static_cast<size_t>(z_sym) * slice_size] : nullptr;

        for (int p_start = 0; p_start < total_projections; p_start += 16) {
            const int p_end   = std::min(p_start + 16, total_projections);
            const int b_count = p_end - p_start;

            const float* b_row_p[16];
            const float* b_row_n[16];
            for (int i = 0; i < b_count; ++i) {
                const int p = p_start + i;
                const auto& bp = all_params[p];
                const int v0_p = static_cast<int>(std::floor(bp.dv_dz * fz     + bp.v_base + 0.5f));
                const int v0_n = static_cast<int>(std::floor(bp.dv_dz * fz_sym + bp.v_base + 0.5f));
                b_row_p[i] = (v0_p >= 0 && v0_p < nh) ? &projs_raw[(p * nh + v0_p) * nw] : nullptr;
                b_row_n[i] = (vol_slice_n && v0_n >= 0 && v0_n < nh) ? &projs_raw[(p * nh + v0_n) * nw] : nullptr;
            }

            for (int y = 0; y < ny; ++y) {
                const float fy = static_cast<float>(y);
                float* vol_row_p = vol_slice_p + static_cast<size_t>(y) * nx;
                float* vol_row_n = vol_slice_n ? vol_slice_n + static_cast<size_t>(y) * nx : nullptr;

                for (int i = 0; i < b_count; ++i) {
                    const float* r_p = b_row_p[i];
                    const float* r_n = b_row_n[i];
                    if (!r_p && !r_n) continue;

                    const int p = p_start + i;
                    const auto& bp = all_params[p];
                    const float du        = bp.du_dx;
                    const float u0_p_base = fy * bp.du_dy + fz     * bp.du_dz + bp.u_base_z0;
                    const float u0_n_base = fy * bp.du_dy + fz_sym * bp.du_dz + bp.u_base_z0;

                    if (r_p) {
                        int x_start = 0, x_end = nx;
                        if (du != 0.0f) {
                            float xs = (0.0f                          - u0_p_base) / du;
                            float xe = (static_cast<float>(nw - 1) - u0_p_base) / du;
                            if (du < 0) std::swap(xs, xe);
                            x_start = std::max(0,  static_cast<int>(std::ceil(xs)));
                            x_end   = std::min(nx, static_cast<int>(std::floor(xe)));
                        }
                        if (x_start < x_end)
                            backproject_row_avx2(vol_row_p, r_p, u0_p_base, du, x_start, x_end);
                    }

                    if (r_n) {
                        int x_start = 0, x_end = nx;
                        if (du != 0.0f) {
                            float xs = (0.0f                          - u0_n_base) / du;
                            float xe = (static_cast<float>(nw - 1) - u0_n_base) / du;
                            if (du < 0) std::swap(xs, xe);
                            x_start = std::max(0,  static_cast<int>(std::ceil(xs)));
                            x_end   = std::min(nx, static_cast<int>(std::floor(xe)));
                        }
                        if (x_start < x_end)
                            backproject_row_avx2(vol_row_n, r_n, u0_n_base, du, x_start, x_end);
                    }
                }
            }
        }
    }
}

void OptimizedBackprojectionCPU::transposeProjections(const Buffer2D&, float*, int, int, int) {}
void OptimizedBackprojectionCPU::backprojectionBatch(Volume&, const float*, const std::vector<ProjectionMatrix>&, int, int, const CTGeometry&) {}

} // namespace ct
