#include "OptimizedBackprojectionCPU.h"
#include <cmath>
#include <algorithm>
#include <omp.h>

namespace ct {

void OptimizedBackprojectionCPU::reconstruct(Volume& volume,
                                            const Buffer2D& projections,
                                            const std::vector<ProjectionMatrix>& matrices,
                                            const CTGeometry& geom,
                                            int batch_size) {
    if (volume.empty() || projections.empty() || matrices.empty()) return;

    int np = geom.np;
    int nw = geom.nw;
    int nh = geom.nh;

    // Separate memory for transposed projections: [np][nw][nh]
    // This allows linear access to detector rows (h) which correspond to slices
    std::vector<float> transposed(static_cast<size_t>(np) * nw * nh);
    transposeProjections(projections, transposed.data(), np, nh, nw);

    // Initialize volume with zeros
    std::fill(volume.data.begin(), volume.data.end(), 0.0f);

    // Batch processing
    backprojectionBatch(volume, transposed.data(), matrices, np, batch_size, geom);
}

void OptimizedBackprojectionCPU::transposeProjections(const Buffer2D& src, 
                                                     float* dst, 
                                                     int np, int nh, int nw) {
    // New layout: [Angle][V_bin][U_bin] (Row-Major)
    // This allows contiguous reads of detector rows during the pre-blending phase.
    // If the source is already row-major [p][h][w], this is effectively a flat copy.
    size_t total_size = static_cast<size_t>(np) * nh * nw;
    std::copy(src.data.begin(), src.data.begin() + total_size, dst);
}

void OptimizedBackprojectionCPU::backprojectionBatch(Volume& volume, 
                                                    const float* transposed_projs,
                                                    const std::vector<ProjectionMatrix>& matrices,
                                                    int total_projections,
                                                    int batch_size,
                                                    const CTGeometry& geom) {
    int nx = geom.nx;
    int ny = geom.ny;
    int nz = geom.nz;
    int nw = geom.nw;
    int nh = geom.nh;

    int nz_half = (nz + 1) / 2;

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

    #pragma omp parallel
    {
        // Thread-local scratch space sized exactly to nw for cache efficiency
        std::vector<float> b_pos_storage(batch_size * nw, 0.0f);
        std::vector<float> b_neg_storage(batch_size * nw, 0.0f);
        float* b_pos_ptr = b_pos_storage.data();
        float* b_neg_ptr = b_neg_storage.data();

        for (int b_start = 0; b_start < total_projections; b_start += batch_size) {
            int b_count = std::min(batch_size, total_projections - b_start);

            #pragma omp for schedule(dynamic)
            for (int z = 0; z < nz_half; ++z) {
                int z_sym = nz - 1 - z;
                bool is_z_symmetry = (z != z_sym);

                for (int b = 0; b < b_count; ++b) {
                    const auto& bp = all_params[b_start + b];
                    const float* p_base = &transposed_projs[(b_start + b) * nw * nh];
                    float* b_row_p = &b_pos_ptr[b * nw];
                    float* b_row_n = &b_neg_ptr[b * nw];

                    // --- Phase 1: Vertical Blending (Inlined for speed) ---
                    float v_p = bp.dv_dz * z + bp.v_base;
                    int v0_p = static_cast<int>(std::floor(v_p));
                    if (v0_p >= 0 && v0_p < nh - 1) {
                        float fv1_p = v_p - static_cast<float>(v0_p);
                        float fv0_p = 1.0f - fv1_p;
                        const float* r0 = &p_base[v0_p * nw];
                        const float* r1 = &p_base[(v0_p + 1) * nw];
                        #pragma omp simd
                        for (int u = 0; u < nw; ++u) {
                            b_row_p[u] = r0[u] * fv0_p + r1[u] * fv1_p;
                        }
                    } else {
                        std::fill(b_row_p, b_row_p + nw, 0.0f);
                    }

                    if (is_z_symmetry) {
                        float v_n = bp.dv_dz * z_sym + bp.v_base;
                        int v0_n = static_cast<int>(std::floor(v_n));
                        if (v0_n >= 0 && v0_n < nh - 1) {
                            float fv1_n = v_n - static_cast<float>(v0_n);
                            float fv0_n = 1.0f - fv1_n;
                            const float* r0 = &p_base[v0_n * nw];
                            const float* r1 = &p_base[(v0_n + 1) * nw];
                            #pragma omp simd
                            for (int u = 0; u < nw; ++u) {
                                b_row_n[u] = r0[u] * fv0_n + r1[u] * fv1_n;
                            }
                        } else {
                            std::fill(b_row_n, b_row_n + nw, 0.0f);
                        }
                    }
                }

                // --- Phase 2: Horizontal Interpolation & Accumulation ---
                for (int y = 0; y < ny; ++y) {
                    const float fy = static_cast<float>(y);
                    const float fz = static_cast<float>(z);
                    const float fz_s = static_cast<float>(z_sym);
                    float* vol_p = &volume.at(0, y, z);
                    float* vol_n = is_z_symmetry ? &volume.at(0, y, z_sym) : nullptr;

                    float u_starts[64];
                    bool du_dz_is0 = true;
                    for (int b = 0; b < b_count; ++b) {
                        const auto& bp = all_params[b_start + b];
                        u_starts[b] = fy * bp.du_dy + fz * bp.du_dz + bp.u_base_z0;
                        if (bp.du_dz != 0.0f) du_dz_is0 = false;
                    }

                    for (int x = 0; x < nx; ++x) {
                        float s_pos = 0.0f;
                        float s_neg = 0.0f;
                        const float fx = static_cast<float>(x);

                        // Process batch loop (not vectorizing x to preserve record accumulation in registers)
                        for (int b = 0; b < b_count; ++b) {
                            const auto& bp = all_params[b_start + b];
                            float u = fx * bp.du_dx + u_starts[b];
                            int u0 = static_cast<int>(u);
                            
                            if (u0 >= 0 && u0 < nw - 1) {
                                float fu1 = u - static_cast<float>(u0);
                                float fu0 = 1.0f - fu1;
                                float* r_pos = &b_pos_ptr[b * nw];
                                s_pos += r_pos[u0] * fu0 + r_pos[u0 + 1] * fu1;
                                
                                if (vol_n) {
                                    float* r_neg = &b_neg_ptr[b * nw];
                                    if (du_dz_is0) {
                                        s_neg += r_neg[u0] * fu0 + r_neg[u0 + 1] * fu1;
                                    } else {
                                        float u_s = fy * bp.du_dy + fz_s * bp.du_dz + fx * bp.du_dx + bp.u_base_z0;
                                        int u0_s = static_cast<int>(u_s);
                                        if (u0_s >= 0 && u0_s < nw - 1) {
                                            float fu1_s = u_s - static_cast<float>(u0_s);
                                            s_neg += r_neg[u0_s] * (1.0f - fu1_s) + r_neg[u0_s + 1] * fu1_s;
                                        }
                                    }
                                }
                            }
                        }
                        vol_p[x] += s_pos;
                        if (vol_n) vol_n[x] += s_neg;
                    }
                }
            }
        }
    }
}

} // namespace ct
