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
    // Original layout: [np][nh][nw] (implied by Sinogram where height=nh, width=np?)
    // Actually in this project Sinogram.data is [nh][np] or similar.
    // Let's assume src is [np][nh][nw] as a flat array.
    // If it's a Sinogram-like Buffer2D(nw*np, nh), we need to be careful.
    // In our case, we pass reconstructed projections which are usually [np][nh][nw].
    
    #pragma omp parallel for collapse(2)
    for (int p = 0; p < np; ++p) {
        for (int w = 0; w < nw; ++w) {
            for (int h = 0; h < nh; ++h) {
                // dst[p][w][h] = src[p][h][w]
                // Correct indexing for flatten 3D array
                dst[(p * nw + w) * nh + h] = src.data[(p * nh + h) * nw + w];
            }
        }
    }
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

    // Pre-calculate all projection parameters once to avoid any overhead in parallel sections
    struct ProjParams {
        float u_base_z0;
        float du_dx, du_dy, du_dz;
        float v_base;
        float dv_dz;
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
        // Persistent threat-local workspace (allocated ONCE per reconstruction)
        std::vector<float> b_pos_storage(batch_size * 2048, 0.0f);
        std::vector<float> b_neg_storage(batch_size * 2048, 0.0f);
        float (*blended_pos)[2048] = reinterpret_cast<float(*)[2048]>(b_pos_storage.data());
        float (*blended_neg)[2048] = reinterpret_cast<float(*)[2048]>(b_neg_storage.data());

        for (int b_start = 0; b_start < total_projections; b_start += batch_size) {
            int b_count = std::min(batch_size, total_projections - b_start);

            #pragma omp for schedule(dynamic)
            for (int z = 0; z < nz_half; ++z) {
                int z_sym = nz - 1 - z;
                bool is_z_symmetry = (z != z_sym);

                for (int b = 0; b < b_count; ++b) {
                    const auto& p_idx = b_start + b;
                    const auto& bp = all_params[p_idx];

                    float v_pos = bp.dv_dz * z + bp.v_base;
                    int v0_p = static_cast<int>(std::floor(v_pos));
                    float fv1_p = v_pos - v0_p;
                    float fv0_p = 1.0f - fv1_p;

                    const float* p_base = &transposed_projs[p_idx * nw * nh];
                    float* b_row_p = blended_pos[b];
                    
                    if (v0_p >= 0 && v0_p < nh - 1) {
                        #pragma omp simd
                        for (int u = 0; u < nw; ++u) {
                            b_row_p[u] = p_base[u * nh + v0_p] * fv0_p + p_base[u * nh + v0_p + 1] * fv1_p;
                        }
                    } else {
                        std::fill(b_row_p, b_row_p + nw, 0.0f);
                    }
                    
                    if (is_z_symmetry) {
                        float v_neg = bp.dv_dz * z_sym + bp.v_base;
                        int v0_n = static_cast<int>(std::floor(v_neg));
                        float fv1_n = v_neg - v0_n;
                        float fv0_n = 1.0f - fv1_n;
                        float* b_row_n = blended_neg[b];
                        
                        if (v0_n >= 0 && v0_n < nh - 1) {
                            #pragma omp simd
                            for (int u = 0; u < nw; ++u) {
                                b_row_n[u] = p_base[u * nh + v0_n] * fv0_n + p_base[u * nh + v0_n + 1] * fv1_n;
                            }
                        } else {
                            std::fill(b_row_n, b_row_n + nw, 0.0f);
                        }
                    }
                }

                for (int y = 0; y < ny; ++y) {
                    const float fy = static_cast<float>(y);
                    const float fz = static_cast<float>(z);
                    const float fz_s = static_cast<float>(z_sym);
                    float* vol_row_pos = &volume.at(0, y, z);
                    float* vol_row_neg = is_z_symmetry ? &volume.at(0, y, z_sym) : nullptr;

                    // Hoist projection coordinate start positions for the current row
                    float u_starts[64];
                    float u_sym_starts[64];
                    for (int b = 0; b < b_count; ++b) {
                        const auto& bp = all_params[b_start + b];
                        u_starts[b] = fy * bp.du_dy + fz * bp.du_dz + bp.u_base_z0;
                        if (is_z_symmetry) {
                            u_sym_starts[b] = fy * bp.du_dy + fz_s * bp.du_dz + bp.u_base_z0;
                        }
                    }

                    for (int x = 0; x < nx; ++x) {
                        float s_pos = 0.0f;
                        float s_neg = 0.0f;
                        const float fx = static_cast<float>(x);

                        #pragma omp simd reduction(+:s_pos, s_neg)
                        for (int b = 0; b < b_count; ++b) {
                            const float du = all_params[b_start + b].du_dx;
                            
                            float u = fx * du + u_starts[b];
                            int u0 = static_cast<int>(u); // Fast cast for positive range
                            if (u0 >= 0 && u0 < nw - 1) {
                                float fu1 = u - u0;
                                s_pos += blended_pos[b][u0] * (1.0f - fu1) + blended_pos[b][u0 + 1] * fu1;
                            }

                            if (is_z_symmetry) {
                                float u_s = fx * du + u_sym_starts[b];
                                int u0_s = static_cast<int>(u_s);
                                if (u0_s >= 0 && u0_s < nw - 1) {
                                    float fu1_s = u_s - u0_s;
                                    s_neg += blended_neg[b][u0_s] * (1.0f - fu1_s) + blended_neg[b][u0_s + 1] * fu1_s;
                                }
                            }
                        }
                        vol_row_pos[x] += s_pos;
                        if (is_z_symmetry) vol_row_neg[x] += s_neg;
                    }
                }
            }
        }
    }
}

} // namespace ct
