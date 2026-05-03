#include "OptimizedBackprojectionCPU.h"
#include <cmath>
#include <algorithm>
#include <omp.h>

namespace ct {

// Ширина шага разворота по X в process_row совпадает с шириной SIMD-регистра:
//   AVX-512: 16 float32 — один ZMM
//   AVX2:     8 float32 — один YMM
//   SSE/scalar: 4 float32 — один XMM
// Компилятор разворачивает основной цикл по x в ровно одну SIMD-инструкцию.
#if defined(__AVX512F__)
    constexpr int BP_SIMD_WIDTH = 16;
#elif defined(__AVX2__) || defined(__AVX__)
    constexpr int BP_SIMD_WIDTH = 8;
#else
    constexpr int BP_SIMD_WIDTH = 4;
#endif

void OptimizedBackprojectionCPU::reconstruct(Volume& volume,
                                            const Buffer2D& projections,
                                            const std::vector<ProjectionMatrix>& matrices,
                                            const CTGeometry& geom,
                                            int /*batch_size_ignored*/) {
    if (volume.empty() || projections.empty() || matrices.empty()) return;

    const int nx = geom.nx;
    const int ny = geom.ny;
    const int nz = geom.nz;
    const int nw = geom.nw;
    const int nh = geom.nh;
    const int total_projections = static_cast<int>(matrices.size());
    const int nz_half = (nz + 1) / 2;  // перемещено выше для NUMA-инициализации

    // [P2] NUMA-aware first-touch: инициализируем буфер объёма нулями
    // в том же распределении потоков, которое будет использоваться при backprojection.
    // После std::vector::resize страницы «не тронуты»; первый запис-доступ
    // маппирует страницу на NUMA-узел того потока, который его сделал.
    // Если инициализировать через std::fill в главном потоке — все страницы
    // окажутся на NUMA-узле 0, и остальные потоки будут читать через межузловую шину.
    //
    // Здесь мы явно параллельно заполняем нулями блоки, соответствующие
    // half-z-слоям (schedule(dynamic) — как в основном цикле), чтобы
    // маппинг страниц совпал с паттерном доступа при реконструкции.
    {
        const size_t vol_stride = static_cast<size_t>(nx) * ny;
        float* raw = volume.data.data();
        #pragma omp parallel for schedule(dynamic)
        for (int z = 0; z < nz_half; ++z) {
            const size_t z_pos     = static_cast<size_t>(z)           * vol_stride;
            const size_t z_sym_pos = static_cast<size_t>(nz - 1 - z)  * vol_stride;
            std::fill(raw + z_pos,     raw + z_pos     + vol_stride, 0.0f);
            if (z != nz - 1 - z)
                std::fill(raw + z_sym_pos, raw + z_sym_pos + vol_stride, 0.0f);
        }
    }


    struct ProjParams {
        float u_base_z0;
        float du_dx, du_dy, du_dz;
        float v_base, dv_dz;
    };
    std::vector<ProjParams> all_params(total_projections);
    for (int p = 0; p < total_projections; ++p) {
        const auto& m = matrices[p].data;
        all_params[p].du_dx   = m[0][0];
        all_params[p].du_dy   = m[0][1];
        all_params[p].du_dz   = m[0][2];
        all_params[p].u_base_z0 = m[0][3];
        all_params[p].dv_dz   = m[1][2];
        all_params[p].v_base  = m[1][3];
    }

    const float* projs_raw = projections.data.data();
    const size_t slice_size = static_cast<size_t>(nx) * ny;

    // BatchData: per-projection data that depends only on Z (not on Y).
    // Pre-computed once per Z-slice, reused for all Y rows.
    struct BatchData {
        const float* r_p;
        const float* r_n;
        float du, u_base_z_p, u_base_z_n, du_dy;
    };

    #pragma omp parallel for schedule(dynamic)
    for (int z = 0; z < nz_half; ++z) {
        const int z_sym = nz - 1 - z;
        const bool has_sym = (z != z_sym);
        const float fz      = static_cast<float>(z);
        const float fz_sym  = static_cast<float>(z_sym);

        float* vol_slice_p = &volume.data[static_cast<size_t>(z)     * slice_size];
        float* vol_slice_n = has_sym ? &volume.data[static_cast<size_t>(z_sym) * slice_size] : nullptr;

        // Pre-compute all per-projection data for this Z once.
        std::vector<BatchData> bd(total_projections);
        for (int p = 0; p < total_projections; ++p) {
            const auto& bp = all_params[p];
            const int v0_p = static_cast<int>(std::floor(bp.dv_dz * fz     + bp.v_base + 0.5f));
            const int v0_n = static_cast<int>(std::floor(bp.dv_dz * fz_sym + bp.v_base + 0.5f));

            bd[p].r_p       = (v0_p >= 0 && v0_p < nh) ? &projs_raw[(p * nh + v0_p) * nw] : nullptr;
            bd[p].r_n       = (vol_slice_n && v0_n >= 0 && v0_n < nh) ? &projs_raw[(p * nh + v0_n) * nw] : nullptr;
            bd[p].du        = bp.du_dx;
            bd[p].du_dy     = bp.du_dy;
            bd[p].u_base_z_p = fz     * bp.du_dz + bp.u_base_z0;
            bd[p].u_base_z_n = fz_sym * bp.du_dz + bp.u_base_z0;
        }

        auto process_row = [&](float* row, const float* proj_row, float u_base, float du) {
            if (!proj_row) return;
            int x_start = 0, x_end = nx;
            if (du != 0.0f) {
                float xs = (0.0f                          - u_base) / du;
                float xe = (static_cast<float>(nw - 1) - u_base) / du;
                if (du < 0) std::swap(xs, xe);
                x_start = std::max(0,  static_cast<int>(std::ceil(xs)));
                x_end   = std::min(nx, static_cast<int>(std::floor(xe)));
            }
            if (x_start >= x_end) return;

            // Основной цикл: BP_SIMD_WIDTH пикселей за итерацию.
            // Компилятор авто-векторизует блок BP_SIMD_WIDTH = ширине регистра.
            int x = x_start;
            for (; x <= x_end - BP_SIMD_WIDTH; x += BP_SIMD_WIDTH) {
                // Развёртка вручную: BP_SIMD_WIDTH независимых acc за итерацию.
                // Значения u0f..u(N-1)f вычисляются без зависимостей → SIMD.
                #pragma omp simd
                for (int d = 0; d < BP_SIMD_WIDTH; ++d) {
                    const float uf = static_cast<float>(x + d) * du + u_base;
                    const int   u0 = static_cast<int>(uf);
                    row[x + d] += proj_row[u0] + (uf - static_cast<float>(u0))
                                  * (proj_row[u0 + 1] - proj_row[u0]);
                }
            }
            // Хвостовой цикл: оставшиеся пиксели (< BP_SIMD_WIDTH)
            // MSVC требует полной формы for (int ...) для #pragma omp simd
            const int x_tail = x;
            #pragma omp simd
            for (int xt = x_tail; xt < x_end; ++xt) {
                const float uf = static_cast<float>(xt) * du + u_base;
                const int u0 = static_cast<int>(uf);
                row[xt] += proj_row[u0] + (uf - static_cast<float>(u0))
                          * (proj_row[u0 + 1] - proj_row[u0]);
            }
        };

        // Iterate projection batches (outer) then Y rows (inner).
        // Each projection row is read exactly once per Z-slice.
        for (int p_start = 0; p_start < total_projections; p_start += 16) {
            const int p_end = std::min(p_start + 16, total_projections);

            for (int y = 0; y < ny; ++y) {
                const float fy = static_cast<float>(y);
                float* vol_row_p = vol_slice_p + static_cast<size_t>(y) * nx;
                float* vol_row_n = vol_slice_n ? vol_slice_n + static_cast<size_t>(y) * nx : nullptr;

                for (int p = p_start; p < p_end; ++p) {
                    const auto& b = bd[p];
                    if (!b.r_p && !b.r_n) continue;

                    const float u_p = fy * b.du_dy + b.u_base_z_p;
                    const float u_n = fy * b.du_dy + b.u_base_z_n;

                    process_row(vol_row_p, b.r_p, u_p, b.du);
                    process_row(vol_row_n, b.r_n, u_n, b.du);
                }
            }
        }
    }
}

void OptimizedBackprojectionCPU::transposeProjections(const Buffer2D&, float*, int, int, int) {}
void OptimizedBackprojectionCPU::backprojectionBatch(Volume&, const float*, const std::vector<ProjectionMatrix>&, int, int, const CTGeometry&) {}

} // namespace ct
