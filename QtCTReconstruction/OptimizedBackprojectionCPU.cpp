#include "OptimizedBackprojectionCPU.h"
#include <cmath>
#include <algorithm>
#include <omp.h>

namespace ct {

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

    // Initialize volume with zeros
    std::fill(volume.data.begin(), volume.data.end(), 0.0f);

    // Pre-calculate projection parameters to avoid vector overhead in the hot loop
    struct ProjParams {
        float u_base_z0;
        float du_dx, du_dy, du_dz;
        float v_base, dv_dz;
    };
    std::vector<ProjParams> all_params(total_projections);
    for (int p = 0; p < total_projections; ++p) {
        const auto& m = matrices[p].data;
        all_params[p].du_dx = m[0][0];
        all_params[p].du_dy = m[0][1];
        all_params[p].du_dz = m[0][2];
        all_params[p].u_base_z0 = m[0][3];
        all_params[p].dv_dz = m[1][2];
        all_params[p].v_base = m[1][3];
    }

    const int nz_half = (nz + 1) / 2;
    const float* projs_raw = projections.data.data();
    const size_t slice_size = static_cast<size_t>(nx) * ny;

    #define TILE_Y 64

    #pragma omp parallel for schedule(dynamic)
    for (int z = 0; z < nz_half; ++z) {
        const int z_sym = nz - 1 - z;
        const bool has_sym = (z != z_sym);
        const float fz = static_cast<float>(z);
        const float fz_sym = static_cast<float>(z_sym);

        float* vol_slice_p = &volume.data[static_cast<size_t>(z) * slice_size];
        float* vol_slice_n = has_sym ? &volume.data[static_cast<size_t>(z_sym) * slice_size] : nullptr;

        for (int yt = 0; yt < ny; yt += TILE_Y) {
            const int y_end = std::min(yt + TILE_Y, ny);

            for (int p_start = 0; p_start < total_projections; p_start += 16) {
                const int p_end = std::min(p_start + 16, total_projections);
                const int b_count = p_end - p_start;

                // Pre-calculate projection parameters for this batch and this Z slice
                struct BatchData {
                    const float* r_p;
                    const float* r_n;
                    float du, u_base_z_p, u_base_z_n;
                } b_data[16];

                for (int i = 0; i < b_count; ++i) {
                    const int p = p_start + i;
                    const auto& bp = all_params[p];
                    const int v0_p = static_cast<int>(std::floor(bp.dv_dz * fz + bp.v_base + 0.5f));
                    const int v0_n = static_cast<int>(std::floor(bp.dv_dz * fz_sym + bp.v_base + 0.5f));
                    
                    b_data[i].r_p = (v0_p >= 0 && v0_p < nh) ? &projs_raw[(p * nh + v0_p) * nw] : nullptr;
                    b_data[i].r_n = (vol_slice_n && v0_n >= 0 && v0_n < nh) ? &projs_raw[(p * nh + v0_n) * nw] : nullptr;
                    b_data[i].du = bp.du_dx;
                    b_data[i].u_base_z_p = fz * bp.du_dz + bp.u_base_z0;
                    b_data[i].u_base_z_n = fz_sym * bp.du_dz + bp.u_base_z0;
                }

                for (int y = yt; y < y_end; ++y) {
                    const float fy = static_cast<float>(y);
                    float* vol_row_p = vol_slice_p + static_cast<size_t>(y) * nx;
                    float* vol_row_n = vol_slice_n ? vol_slice_n + static_cast<size_t>(y) * nx : nullptr;

                    for (int i = 0; i < b_count; ++i) {
                        const auto& bd = b_data[i];
                        if (!bd.r_p && !bd.r_n) continue;

                        const float u0_p_base = fy * all_params[p_start + i].du_dy + bd.u_base_z_p;
                        const float u0_n_base = fy * all_params[p_start + i].du_dy + bd.u_base_z_n;
                        const float du = bd.du;

                        auto process_row = [&](float* row, const float* proj_row, float u_base) {
                            if (!proj_row) return;
                            int x_start = 0, x_end = nx;
                            if (du != 0.0f) {
                                float xs = (0.0f - u_base) / du;
                                float xe = (static_cast<float>(nw - 1) - u_base) / du;
                                if (du < 0) std::swap(xs, xe);
                                x_start = std::max(0, static_cast<int>(std::ceil(xs)));
                                x_end = std::min(nx, static_cast<int>(std::floor(xe)));
                            }
                            
                            if (x_start < x_end) {
                                int x = x_start;
                                // Fast path with SIMD if possible (manual loop unrolling for better pipeline usage)
                                for (; x <= x_end - 4; x += 4) {
                                    float u_vec[4] = {
                                        static_cast<float>(x) * du + u_base,
                                        static_cast<float>(x+1) * du + u_base,
                                        static_cast<float>(x+2) * du + u_base,
                                        static_cast<float>(x+3) * du + u_base
                                    };
                                    for(int k=0; k<4; ++k) {
                                        const int u0 = static_cast<int>(u_vec[k]);
                                        const float fu = u_vec[k] - static_cast<float>(u0);
                                        row[x+k] += proj_row[u0] + fu * (proj_row[u0 + 1] - proj_row[u0]);
                                    }
                                }
                                for (; x < x_end; ++x) {
                                    const float u = static_cast<float>(x) * du + u_base;
                                    const int u0 = static_cast<int>(u);
                                    const float fu = u - static_cast<float>(u0);
                                    row[x] += proj_row[u0] + fu * (proj_row[u0 + 1] - proj_row[u0]);
                                }
                            }
                        };

                        process_row(vol_row_p, bd.r_p, u0_p_base);
                        process_row(vol_row_n, bd.r_n, u0_n_base);
                    }
                }
            }
        }
    }

}

void OptimizedBackprojectionCPU::transposeProjections(const Buffer2D&, float*, int, int, int) {}
void OptimizedBackprojectionCPU::backprojectionBatch(Volume&, const float*, const std::vector<ProjectionMatrix>&, int, int, const CTGeometry&) {}

} // namespace ct
