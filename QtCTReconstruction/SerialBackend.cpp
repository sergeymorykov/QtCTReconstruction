// SerialBackend.cpp — эталонная последовательная CPU-реализация.
//
// Намеренно использует только:
//   - простые for-циклы
//   - std::sin / std::cos
//   - FFT::forwardReal / FFT::inverseReal (учебный Cooley–Tukey из FFT.cpp)
//
// НЕТ: #pragma omp, __m256, _mm256_*, cudaXxx.
//
// Соответствие формулам статьи:
//   computeSinogram       -> формула (2):  p(theta, s) = sum_{x,y} f(x,y) * delta_bilinear
//   filterProjection      -> формула (5):  q(theta, s) = p(theta, .) * h_ramp  (свёртка через FFT)
//   backproject           -> формулы (6)-(7): f_recon(x,y) = (pi/N_theta) * sum_theta q(theta, x cos theta - y sin theta)
//   reconstructSlice      -> формулы (5)-(8) целиком
//   reconstructVolume     -> последовательный цикл по срезам z

#include "SerialBackend.h"

#include "FFT.h"
#include "FilterDesign.h"
#include "Utils.h"

#include <cmath>
#include <algorithm>
#include <limits>
#include <chrono>

namespace ct {

// ============================================================
// Внутренние вспомогательные функции (только в этом .cpp)
// ============================================================
namespace {

// Формула (2): вклад пикселя (x, y) в синограмму методом линейной интерполяции.
// theta — угол проецирования [рад], cx/cy — центры среза.
// Возвращает вещественную координату s на детекторе.
inline float projectPixel(float x, float y, float cos_th, float sin_th,
                           float cx, float cy, float detector_center) noexcept {
    // s = (x - cx) * cos(theta) + (y - cy) * sin(theta) + detector_center
    return (x - cx) * cos_th + (y - cy) * sin_th + detector_center;
}

// Линейная интерполяция значения в проекции по вещественному индексу u.
inline float sampleLinear(const float* row, int n_bins, float u) noexcept {
    const int i0 = static_cast<int>(std::floor(u));
    const int i1 = i0 + 1;
    const float frac = u - static_cast<float>(i0);
    if (i0 < 0 || i1 >= n_bins) return 0.0f;
    return row[i0] * (1.0f - frac) + row[i1] * frac;
}

} // anonymous namespace


// ============================================================
// Формула (2): преобразование Радона — прямое проецирование среза
// ============================================================
Sinogram SerialBackend::computeSinogram(const Buffer2D& slice,
                                        size_t num_angles,
                                        size_t detector_bins,
                                        bool /*use_parallel — игнорируется в serial-версии*/) {
    Sinogram sino;
    if (slice.empty() || num_angles == 0 || detector_bins == 0) return sino;

    const size_t w = slice.width;
    const size_t h = slice.height;

    // Нахождение диапазона HU: последовательный проход
    float hu_min = std::numeric_limits<float>::max();
    float hu_max = std::numeric_limits<float>::lowest();
    for (size_t idx = 0; idx < slice.data.size(); ++idx) {
        const float v = slice.data[idx];
        if (v < hu_min) hu_min = v;
        if (v > hu_max) hu_max = v;
    }

    float span = hu_max - hu_min;
    if (span < 1e-6f) span = 1.0f;
    const float inv_span = 1.0f / span;

    // Аллокация синограммы: layout [num_angles][detector_bins]
    sino.data.assign(detector_bins, num_angles, 0.0f);

    const float cx = static_cast<float>(w - 1) * 0.5f;
    const float cy = static_cast<float>(h - 1) * 0.5f;
    const float detector_center = static_cast<float>(detector_bins / 2);
    const int db = static_cast<int>(detector_bins);

    // Формула (2): для каждого угла theta — scatter-проекция
    for (size_t a = 0; a < num_angles; ++a) {
        const float th = utils::degToRad(
            180.0f / static_cast<float>(num_angles) * static_cast<float>(a));
        const float cos_th = std::cos(th);
        const float sin_th = std::sin(th);
        float* sino_row = sino.data[a];

        for (size_t y = 0; y < h; ++y) {
            const float* src_row = &slice.data[y * w];
            const float yy = static_cast<float>(y);

            for (size_t x = 0; x < w; ++x) {
                const float v_raw = src_row[x];
                if (v_raw == 0.0f) continue;               // пропуск фона

                const float v = (v_raw - hu_min) * inv_span;
                const float xx = static_cast<float>(x);

                // s = (x - cx)*cos + (y - cy)*sin + detector_center
                const float s = (xx - cx) * cos_th + (yy - cy) * sin_th + detector_center;
                const int i0 = static_cast<int>(std::floor(s));

                // Линейная интерполяция (bilinear scatter)
                if (i0 >= 0 && i0 < db - 1) {
                    const float frac = s - static_cast<float>(i0);
                    sino_row[i0]     += v * (1.0f - frac);
                    sino_row[i0 + 1] += v * frac;
                } else if (i0 == db - 1) {
                    sino_row[i0] += v;
                }
            }
        }
    }

    // Заполнение метаданных синограммы
    sino.angles_deg.resize(num_angles);
    const float angle_step = 180.0f / static_cast<float>(num_angles);
    for (size_t a = 0; a < num_angles; ++a) {
        sino.angles_deg[a] = angle_step * static_cast<float>(a);
    }
    sino.detector_spacing_mm = 1.0f;
    sino.original_min_hu     = hu_min;
    sino.original_max_hu     = hu_max;

    return sino;
}


// ============================================================
// Формулы (5)-(8): FBP — фильтрация + обратная проекция
// ============================================================
Buffer2D SerialBackend::reconstructSlice(const Sinogram& sinogram,
                                          size_t output_size,
                                          const ReconstructionParams& params) {
    const size_t detector_bins = sinogram.data.width;
    const size_t num_angles    = sinogram.data.height;

    if (detector_bins == 0 || num_angles == 0 || output_size == 0) return {};

    // --- Шаг 1: расширение детектора до квадрата (диагональ slice) ---
    // Формула (5): дополнение нулями до длины ceil(sqrt(2) * detector_bins)
    const size_t square_bins = static_cast<size_t>(
        std::ceil(std::sqrt(2.0) * static_cast<double>(detector_bins)));
    const size_t old_center = detector_bins / 2;
    const size_t new_center = square_bins / 2;
    const size_t pad_before = new_center - old_center;

    // --- Шаг 2: предвычисление рамп-фильтра (формула (5)) ---
    const size_t padding_factor = 2;
    const size_t fft_size = std::max<size_t>(
        64, utils::nextPowerOfTwo(padding_factor * square_bins));
    const auto filter = FilterDesign::createFilter(fft_size, params.filter);
    const size_t spectrum_size = fft_size / 2 + 1;

    // Рабочие буферы (переиспользуются для каждого угла)
    std::vector<float>   proj_padded(fft_size, 0.0f);
    std::vector<Complex> spectrum(spectrum_size);
    std::vector<float>   q(fft_size);

    // --- Шаг 3: фильтрация каждой проекции (формула (5)) ---
    // filtered[a][i] = обратное FFT(FFT(p[a]) * H)
    Buffer2D filtered(square_bins, num_angles, 0.0f);

    for (size_t a = 0; a < num_angles; ++a) {
        // Копируем проекцию с паддингом до square_bins
        std::fill(proj_padded.begin(), proj_padded.end(), 0.0f);
        const float* src_row = sinogram.data[a];
        for (size_t i = 0; i < detector_bins; ++i) {
            const size_t dst = i + pad_before;
            if (dst < fft_size) proj_padded[dst] = src_row[i];
        }

        // FFT -> умножение на фильтр -> IFFT  (формула (5))
        FFT::forwardReal(proj_padded, spectrum);
        for (size_t k = 0; k < spectrum_size; ++k) {
            spectrum[k] *= filter[k];
        }
        FFT::inverseReal(spectrum, q);

        // Сохраняем square_bins отсчётов
        float* dst_row = filtered[a];
        for (size_t i = 0; i < square_bins; ++i) {
            dst_row[i] = q[i];
        }
    }

    // --- Шаг 4: предвычисление тригонометрии ---
    std::vector<float> cos_table(num_angles);
    std::vector<float> sin_table(num_angles);
    for (size_t a = 0; a < num_angles; ++a) {
        const float th = utils::degToRad(sinogram.angles_deg[a]);
        cos_table[a] = std::cos(th);
        sin_table[a] = std::sin(th);
    }

    // --- Шаг 5: обратная проекция (формулы (6)-(7)) ---
    // f_recon(x,y) = sum_theta q(theta, x*cos(theta) - y*sin(theta))
    Slice recon(output_size, output_size, 0.0f);
    const float radius          = static_cast<float>(output_size / 2);
    const float detector_center = static_cast<float>(square_bins / 2);
    const int   sq_bins_int     = static_cast<int>(square_bins);

    for (size_t y = 0; y < output_size; ++y) {
        const float yy = static_cast<float>(y) - radius;
        float* recon_row = recon[y];

        for (size_t x = 0; x < output_size; ++x) {
            const float xx = static_cast<float>(x) - radius;

            // Пиксели вне вписанного круга не реконструируются
            if (xx * xx + yy * yy > radius * radius) {
                recon_row[x] = 0.0f;
                continue;
            }

            float sum = 0.0f;
            for (size_t a = 0; a < num_angles; ++a) {
                // t = x*cos(theta) - y*sin(theta)  (формула (6))
                const float t = xx * cos_table[a] - yy * sin_table[a];
                const float u = t + detector_center;

                // Линейная интерполяция отфильтрованной проекции
                const int i0 = static_cast<int>(std::floor(u));
                const int i1 = i0 + 1;
                const float frac = u - static_cast<float>(i0);
                if (i0 >= 0 && i1 < sq_bins_int) {
                    const float* f_row = filtered[a];
                    sum += f_row[i0] * (1.0f - frac) + f_row[i1] * frac;
                }
            }
            recon_row[x] = sum;
        }
    }

    // --- Шаг 6: нормировка (формула (8)): умножение на pi / (2 * N_theta) ---
    const float scale = 3.14159265358979323846f / (2.0f * static_cast<float>(num_angles));
    for (size_t y = 0; y < output_size; ++y) {
        float* row = recon[y];
        for (size_t x = 0; x < output_size; ++x) {
            row[x] *= scale;
        }
    }

    return recon;
}


// ============================================================
// Последовательная реконструкция тома: срез за срезом
// ============================================================
void SerialBackend::reconstructVolume(const Volume& input_volume,
                                       Volume& out_reconstruction,
                                       const ReconstructionParams& params,
                                       std::function<void(int, const Buffer2D&)> onSliceDone) {
    if (input_volume.empty()) return;

    const size_t depth  = input_volume.depth;
    const size_t width  = input_volume.width;
    const size_t height = input_volume.height;

    out_reconstruction.assign(width, height, depth, 0.0f);
    out_reconstruction.x_coords = input_volume.x_coords;
    out_reconstruction.y_coords = input_volume.y_coords;
    out_reconstruction.z_coords = input_volume.z_coords;

    // Раздельное накопление времени: синограмма и реконструкция считаются отдельно,
    // чтобы контроллер мог отобразить их как два независимых показателя.
    double sino_ms  = 0.0;
    double recon_ms = 0.0;

    for (size_t z = 0; z < depth; ++z) {
        // Извлекаем срез
        const Buffer2D original_slice = input_volume.getSlice(z);

        // --- Прямое проецирование (формула (2)) ---
        auto t0 = std::chrono::steady_clock::now();
        const Sinogram sino = computeSinogram(
            original_slice, params.num_angles, width, /*use_parallel=*/false);
        auto t1 = std::chrono::steady_clock::now();
        sino_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

        // --- FBP (формулы (5)-(8)) ---
        Buffer2D recon_norm = reconstructSlice(sino, width, params);
        auto t2 = std::chrono::steady_clock::now();
        recon_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();

        // Денормализация в единицы HU
        const float span     = sino.original_max_hu - sino.original_min_hu;
        const float hu_min   = sino.original_min_hu;
        const float eff_span = (span <= 0.0f) ? 1.0f : span;

        for (size_t i = 0; i < recon_norm.data.size(); ++i) {
            recon_norm.data[i] = recon_norm.data[i] * eff_span + hu_min;
        }

        out_reconstruction.setSlice(z, recon_norm);

        if (onSliceDone) {
            onSliceDone(static_cast<int>(z), recon_norm);
        }
    }

    // Контроллер использует lastSinogramTimeMs() для поля "Sinogram",
    // а reconTimeSec = total - sinogramTimeSec.
    // Чтобы оба поля отображались корректно, сохраняем только sino_ms,
    // а total покроет sino + recon.
    m_lastSinogramTimeMs = sino_ms;
}



// ============================================================
// Извлечение облака точек по порогу HU (последовательный проход)
// ============================================================
PointCloud SerialBackend::extractPointCloud(const Volume& vol, float threshold) {
    if (vol.empty()) return {};

    PointCloud cloud;
    for (size_t z = 0; z < vol.depth; ++z) {
        const float z_coord = vol.z_coords.empty()
            ? static_cast<float>(z) : vol.z_coords[z];
        for (size_t y = 0; y < vol.height; ++y) {
            const float y_coord = vol.y_coords.empty()
                ? static_cast<float>(y) : vol.y_coords[y];
            for (size_t x = 0; x < vol.width; ++x) {
                const float hu = vol.at(x, y, z);
                if (hu > threshold) {
                    const float x_coord = vol.x_coords.empty()
                        ? static_cast<float>(x) : vol.x_coords[x];
                    cloud.push_back({x_coord, y_coord, z_coord, hu});
                }
            }
        }
    }
    return cloud;
}

} // namespace ct
