#include "AppWindow.h"
#include "FileIO.h"
#include "FilteredBackprojection.h"
#include "Generator3D.h"
#include "RadonTransform.h"
#include "Utils.h"

#include <Windows.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <omp.h>
#include <string>
#include <utility>
#include <vector>

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE, PWSTR, int nCmdShow) {
    ct::utils::ensureDirectory(L"data");
    ct::utils::ensureDirectory(L"data\\output");

    ct::Volume volume;
    const std::string input_npy = "data/output/synthetic_brain_hu.npy";
    const std::string input_coords_npy = "data/output/synthetic_brain_hu_coords.npy";
    const bool has_python_coords = static_cast<bool>(std::ifstream(input_coords_npy, std::ios::binary));
    if (!ct::FileIO::loadVolumeNPY(input_npy, volume, has_python_coords)) {
        ct::Generator3D::Params gen_params;
        gen_params.shape = {512, 512, 512};
        gen_params.num_ellipsoids = 300;
        volume = ct::Generator3D::generateBrainHU(gen_params);
        ct::FileIO::saveVolumeNPY(volume, "data/output/synthetic_brain_hu_cxx.npy");
    }

    ct::ReconstructionParams params;
    params.filter = ct::ReconstructionParams::FilterType::Hamming;
    params.num_angles = 360;
    params.zero_padding = true;

    std::vector<ct::Slice> originals = volume.data;
    const size_t depth = volume.depth();
    std::vector<ct::Slice> reconstructions(depth);
    std::vector<ct::Slice> differences(depth);
    std::vector<ct::Sinogram> sinograms(depth);

    const auto t_start = std::chrono::steady_clock::now();

    const int nz = static_cast<int>(depth);
    #pragma omp parallel for schedule(dynamic)
    for (int z = 0; z < nz; ++z) {
        const auto zi = static_cast<size_t>(z);
        const ct::Slice& original = volume.data[zi];
        float hu_min = std::numeric_limits<float>::max();
        float hu_max = std::numeric_limits<float>::lowest();
        for (const auto& row : original) {
            for (const float v : row) {
                hu_min = std::min(hu_min, v);
                hu_max = std::max(hu_max, v);
            }
        }

        ct::Slice normalized = ct::utils::createSlice(original.size(), original[0].size(), 0.0f);
        if (hu_max > hu_min) {
            const float inv_span = 1.0f / (hu_max - hu_min);
            for (size_t y = 0; y < original.size(); ++y) {
                for (size_t x = 0; x < original[y].size(); ++x) {
                    normalized[y][x] = (original[y][x] - hu_min) * inv_span;
                }
            }
        }

        const size_t detector_bins = original.size();
        ct::Sinogram sinogram = ct::RadonTransform::forward(normalized, params.num_angles, detector_bins);
        ct::Slice recon_normalized = ct::FilteredBackprojection::reconstruct(sinogram, original.size(), params);
        ct::Slice reconstruction = ct::utils::createSlice(original.size(), original[0].size(), hu_min);
        if (hu_max > hu_min) {
            const float span = (hu_max - hu_min);
            for (size_t y = 0; y < reconstruction.size(); ++y) {
                for (size_t x = 0; x < reconstruction[y].size(); ++x) {
                    reconstruction[y][x] = recon_normalized[y][x] * span + hu_min;
                }
            }
        }

        differences[zi] = ct::utils::subtract(reconstruction, original);
        sinograms[zi] = std::move(sinogram);
        reconstructions[zi] = std::move(reconstruction);
    }

    const size_t mid_z = volume.depth() / 2;

    ct::FileIO::saveSliceBMP(originals[mid_z], L"data/output/original_middle.bmp", -1000.0f, 100.0f);
    ct::FileIO::saveSliceBMP(reconstructions[mid_z], L"data/output/reconstruction_middle.bmp", -1000.0f, 100.0f);
    ct::FileIO::saveSliceBMP(differences[mid_z], L"data/output/difference_middle.bmp", -200.0f, 200.0f);

    // Central-slice quality metrics for quick parity checks vs Python output.
    double mse = 0.0;
    size_t cnt = 0;
    float vmin = std::numeric_limits<float>::max();
    float vmax = std::numeric_limits<float>::lowest();
    const int metric_rows = static_cast<int>(originals[mid_z].size());
    #pragma omp parallel
    {
        double local_mse = 0.0;
        size_t local_cnt = 0;
        float local_vmin = std::numeric_limits<float>::max();
        float local_vmax = std::numeric_limits<float>::lowest();
        #pragma omp for nowait
        for (int y = 0; y < metric_rows; ++y) {
            const auto yi = static_cast<size_t>(y);
            for (size_t x = 0; x < originals[mid_z][yi].size(); ++x) {
                const double d = static_cast<double>(reconstructions[mid_z][yi][x]) - static_cast<double>(originals[mid_z][yi][x]);
                local_mse += d * d;
                ++local_cnt;
                local_vmin = std::min(local_vmin, originals[mid_z][yi][x]);
                local_vmax = std::max(local_vmax, originals[mid_z][yi][x]);
            }
        }
        #pragma omp critical
        {
            mse += local_mse;
            cnt += local_cnt;
            vmin = std::min(vmin, local_vmin);
            vmax = std::max(vmax, local_vmax);
        }
    }
    mse = (cnt > 0) ? (mse / static_cast<double>(cnt)) : 0.0;
    const double range = static_cast<double>(vmax - vmin);
    const double psnr = (mse <= 0.0 || range <= 0.0) ? std::numeric_limits<double>::infinity()
                                                      : 10.0 * std::log10((range * range) / mse);

    const auto t_end = std::chrono::steady_clock::now();
    const double elapsed_sec = std::chrono::duration<double>(t_end - t_start).count();

    std::ofstream metrics("data/output/metrics.txt", std::ios::out | std::ios::trunc);
    if (metrics.is_open()) {
        metrics << std::fixed << std::setprecision(6);
        metrics << "input_npy=" << input_npy << "\n";
        metrics << "mid_z=" << mid_z << "\n";
        metrics << "mse=" << mse << "\n";
        if (std::isfinite(psnr)) {
            metrics << "psnr=" << psnr << "\n";
        } else {
            metrics << "psnr=inf\n";
        }
        metrics << "elapsed_sec=" << elapsed_sec << "\n";
    }
    metrics.close();

    AppWindow app(hInstance, 1200, 900);
    if (!app.create(L"CT Reconstruction (Win32 + GDI+)")) {
        MessageBoxW(nullptr, L"Failed to create window.", L"CTReconstruction", MB_ICONERROR | MB_OK);
        return -1;
    }

    app.updateDataset(originals, reconstructions, differences, sinograms, mid_z);
    app.show(nCmdShow);
    return app.messageLoop();
}
