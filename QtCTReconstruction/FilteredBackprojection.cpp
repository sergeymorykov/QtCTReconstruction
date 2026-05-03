#include "FilteredBackprojection.h"

#include "IFFTBackend.h"       // thread-local FFT-бэкенд (Builtin или FFTW3)
#include "FilterDesign.h"
#include "Utils.h"

#include <cmath>
#include <omp.h>

namespace ct {
namespace {
float sampleProjectionLinear(const Buffer2D& filtered, const size_t angle_idx, const float u, const size_t detector_count) {
    const int i0 = static_cast<int>(std::floor(u));
    const int i1 = i0 + 1;
    const float frac = u - static_cast<float>(i0);
    if (i0 < 0 || i1 >= static_cast<int>(detector_count)) {
        return 0.0f;
    }
    return utils::lerp(filtered.at(angle_idx, static_cast<size_t>(i0)), filtered.at(angle_idx, static_cast<size_t>(i1)), frac);
}
}

Slice FilteredBackprojection::reconstruct(const Sinogram& sinogram, size_t output_size, const ReconstructionParams& params) {
    // New layout: [Angle][Bin] -> Width = detector_bins, Height = num_angles
    const size_t detector_bins = sinogram.data.width;
    const size_t num_angles = sinogram.data.height;

    const size_t square_bins = static_cast<size_t>(std::ceil(std::sqrt(2.0) * static_cast<double>(detector_bins)));
    const size_t old_center = detector_bins / 2;
    const size_t new_center = square_bins / 2;
    const size_t pad_before = new_center - old_center;

    // Fast layout for backprojection: [Angle][SquareBin]
    Buffer2D sino_square(square_bins, num_angles, 0.0f);
    
    #pragma omp parallel for schedule(static)
    for (int a = 0; a < static_cast<int>(num_angles); ++a) {
        float* dst_row = sino_square[static_cast<size_t>(a)];
        const float* src_row = sinogram.data[static_cast<size_t>(a)];
        for (size_t i = 0; i < detector_bins; ++i) {
            const size_t dst = i + pad_before;
            if (dst < square_bins) {
                dst_row[dst] = src_row[i];
            }
        }
    }

    const size_t padding_factor = 2;
    const size_t projection_size_padded = std::max<size_t>(64, utils::nextPowerOfTwo(padding_factor * square_bins));
    const auto filter = FilterDesign::createFilter(projection_size_padded, params.filter);
    const size_t spectrum_size = projection_size_padded / 2 + 1;  // n/2+1 комплексных точек

    Buffer2D filtered(square_bins, num_angles, 0.0f); // width = bins, height = angles
    const int na = static_cast<int>(num_angles);

    // [P2] Thread-local FFT-план: каждый поток получает собственный IFFTBackend.
    // prepare() вызывается один раз при входе в parallel-регион (или при смене fft_size).
    // Буферы proj_padded/spectrum/q также локальны — нет конкуренции за память.
    #pragma omp parallel
    {
        IFFTBackend& fft = IFFTBackend::threadLocal();
        fft.prepare(projection_size_padded); // no-op если размер не изменился

        std::vector<float>   proj_padded(projection_size_padded, 0.0f);
        std::vector<Complex> spectrum(spectrum_size);
        std::vector<float>   q(projection_size_padded);

        #pragma omp for schedule(dynamic)
        for (int a = 0; a < na; ++a) {
            std::fill(proj_padded.begin(), proj_padded.end(), 0.0f);
            const float* src_row = sino_square[static_cast<size_t>(a)];
            for (size_t i = 0; i < square_bins; ++i) {
                proj_padded[i] = src_row[i];
            }

            fft.forwardReal(proj_padded, spectrum);
            for (size_t k = 0; k < spectrum_size; ++k) {
                spectrum[k] *= filter[k];
            }
            fft.inverseReal(spectrum, q);

            float* dst_row = filtered[static_cast<size_t>(a)];
            for (size_t i = 0; i < square_bins; ++i) {
                dst_row[i] = q[i];
            }
        }
    } // end parallel

    Slice recon(output_size, output_size, 0.0f);
    const float radius = static_cast<float>(output_size / 2);
    const float detector_center = static_cast<float>(square_bins / 2);

    std::vector<float> cos_table(num_angles);
    std::vector<float> sin_table(num_angles);
    for (size_t a = 0; a < num_angles; ++a) {
        const float th = utils::degToRad(sinogram.angles_deg[a]);
        cos_table[a] = std::cos(th);
        sin_table[a] = std::sin(th);
    }

    const int ny = static_cast<int>(output_size);
    #pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < ny; ++y) {
        const float yy = static_cast<float>(y) - radius;
        for (size_t x = 0; x < output_size; ++x) {
            const float xx = static_cast<float>(x) - radius;
            float sum = 0.0f;

            if (xx * xx + yy * yy > radius * radius) {
                recon[static_cast<size_t>(y)][x] = 0.0f;
                continue;
            }

            for (size_t a = 0; a < num_angles; ++a) {
                const float t = xx * cos_table[a] - yy * sin_table[a];
                const float u = t + detector_center;
                
                const int i0 = static_cast<int>(std::floor(u));
                const int i1 = i0 + 1;
                const float frac = u - static_cast<float>(i0);
                if (i0 >= 0 && i1 < static_cast<int>(square_bins)) {
                    const float* f_row = filtered[a];
                    sum += utils::lerp(f_row[i0], f_row[i1], frac);
                }
            }
            recon[static_cast<size_t>(y)][x] = sum;
        }
    }

    const float scale = 3.14159265358979323846f / (2.0f * static_cast<float>(num_angles));
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < ny; ++y) {
        float* row = recon[static_cast<size_t>(y)];
        for (size_t x = 0; x < output_size; ++x) {
            row[x] *= scale;
        }
    }
    return recon;
}

}
