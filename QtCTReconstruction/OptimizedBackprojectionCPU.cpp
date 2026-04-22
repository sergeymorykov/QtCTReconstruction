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

    for (int b_start = 0; b_start < total_projections; b_start += batch_size) {
        int b_count = std::min(batch_size, total_projections - b_start);

        #pragma omp parallel for collapse(2)
        for (int z = 0; z < nz_half; ++z) {
            for (int y = 0; y < ny; ++y) {
                int z_sym = nz - 1 - z;
                float* vol_row_pos = &volume.at(0, y, z);
                float* vol_row_neg = &volume.at(0, y, z_sym);

                // For each projection in the batch, pre-calculate the vertically interpolated row
                // This converts 2D (Bilinear) interpolation into 1D (Linear) in the inner loop.
                // Each thread in the parallel loop gets its own buffer (stack or thread-local)
                float pre_blended_lines[64][1024]; // Max batch size 64, max detector size 1024
                // Note: and the indices where valid data starts
                int u_starts[64];
                int u_ends[64];
                float u0_starts[64];
                float du_steps[64];

                for (int b = 0; b < b_count; ++b) {
                    int p_idx = b_start + b;
                    const auto& m = matrices[p_idx].data;

                    // v depends ONLY on Z and angles (parallel beam)
                    float v = m[1][2] * z + m[1][3];
                    float v_s = m[1][2] * z_sym + m[1][3];

                    int v0 = static_cast<int>(std::floor(v));
                    float fv0 = v - v0;
                    float fv1 = 1.0f - fv0;

                    int v0_s = static_cast<int>(std::floor(v_s));
                    float fvs0 = v_s - v0_s;
                    float fvs1 = 1.0f - fvs0;

                    const float* p_data = &transposed_projs[p_idx * nw * nh];
                    
                    // Pre-blend two rows of the projection for POSITIVE Z and NEGATIVE Z
                    // Since U is the same for both, we can store them together or handle them
                    // We'll calculate the sum contribution for this projection:
                    // val_pos = p[u][v0]*fv1 + p[u][v0+1]*fv0
                    // val_neg = p[u][v0_s]*fvs1 + p[u][v0_s+1]*fvs0
                    
                    // To stay SIMD-friendly, let's just pre-blend a combined row
                    // combined[u] = val_pos (if z==z_sym) OR val_pos and val_neg
                    // But we have two volume rows. Let's just do two line buffers.
                }

                // Actually, let's simplify. Standard hoisting is enough for now.
                // Let's use the most efficient approach:
                struct ProjStep {
                    const float* p0;
                    const float* p1;
                    const float* p0_s;
                    const float* p1_s;
                    float fv0, fv1, fvs0, fvs1;
                    float u_base, du;
                };
                ProjStep steps[64];

                for (int b = 0; b < b_count; ++b) {
                    int p_idx = b_start + b;
                    const auto& m = matrices[p_idx].data;
                    
                    float v = m[1][2] * z + m[1][3];
                    float v_s = m[1][2] * z_sym + m[1][3];

                    int v0 = static_cast<int>(std::floor(v));
                    int v0_s = static_cast<int>(std::floor(v_s));
                    
                    steps[b].fv0 = v - v0;
                    steps[b].fv1 = 1.0f - steps[b].fv0;
                    steps[b].fvs0 = v_s - v0_s;
                    steps[b].fvs1 = 1.0f - steps[b].fvs0;

                    const float* p_base = &transposed_projs[p_idx * nw * nh];
                    if (v0 >= 0 && v0 < nh - 1) {
                        steps[b].p0 = p_base + v0;
                        steps[b].p1 = p_base + v0 + 1;
                    } else {
                        steps[b].p0 = nullptr; steps[b].p1 = nullptr;
                    }

                    if (v0_s >= 0 && v0_s < nh - 1) {
                        steps[b].p0_s = p_base + v0_s;
                        steps[b].p1_s = p_base + v0_s + 1;
                    } else {
                        steps[b].p0_s = nullptr; steps[b].p1_s = nullptr;
                    }

                    steps[b].u_base = m[0][1] * y + m[0][2] * z + m[0][3];
                    steps[b].du = m[0][0];
                }

                for (int x = 0; x < nx; ++x) {
                    float s_pos = 0.0f;
                    float s_neg = 0.0f;

                    for (int b = 0; b < b_count; ++b) {
                        float u = steps[b].u_base + x * steps[b].du;
                        int u0 = static_cast<int>(std::floor(u));
                        float fu0 = u - u0;
                        float fu1 = 1.0f - fu0;

                        if (u0 >= 0 && u0 < nw - 1) {
                            int offset0 = u0 * nh;
                            int offset1 = offset0 + nh;

                            if (steps[b].p0) {
                                float val0 = steps[b].p0[offset0] * steps[b].fv1 + steps[b].p1[offset0] * steps[b].fv0;
                                float val1 = steps[b].p0[offset1] * steps[b].fv1 + steps[b].p1[offset1] * steps[b].fv0;
                                s_pos += val0 * fu1 + val1 * fu0;
                            }

                            if (z != z_sym && steps[b].p0_s) {
                                float val0 = steps[b].p0_s[offset0] * steps[b].fvs1 + steps[b].p1_s[offset0] * steps[b].fvs0;
                                float val1 = steps[b].p0_s[offset1] * steps[b].fvs1 + steps[b].p1_s[offset1] * steps[b].fvs0;
                                s_neg += val0 * fu1 + val1 * fu0;
                            }
                        }
                    }
                    vol_row_pos[x] += s_pos;
                    if (z != z_sym) vol_row_neg[x] += s_neg;
                }
            }
        }
    }
}

} // namespace ct
