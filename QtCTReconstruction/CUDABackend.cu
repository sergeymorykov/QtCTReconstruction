#include "CUDABackend.h"

#include "FilterDesign.h"
#include "Utils.h"

#include <cmath>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <cufft.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ct {

namespace {

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        } \
    } while(0)

__global__ void radonForwardKernel(
    const float* __restrict__ slice,
    float* __restrict__ sino,
    const float* __restrict__ angles_deg,
    int width, int height,
    int num_angles, int detector_bins,
    float cx, float cy, float detector_center) 
{
    int angle_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int bin_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (angle_idx >= num_angles || bin_idx >= detector_bins) return;

    float th = angles_deg[angle_idx] * (M_PI / 180.0f);
    float c = cosf(th);
    float s = sinf(th);

    float sum = 0.0f;

    // В прямом проецировании мы интегрируем вдоль луча.
    // Проходим по всему изображению и сбрасываем вес в бины.
    // Это scatter, что не очень эффективно для GPU (нужны atomicAdd).
    // Поэтому инвертируем цикл для каждого луча. 
    // Для каждого (angle, bin) считаем луч и берем выборки из среза (line integral).
    
    // Вращенные координаты
    // t - вдоль детектора, p - перпендикулярно детектору
    // Нам нужен луч, где t = bin_idx - detector_center
    float t = (float)bin_idx - detector_center;
    
    // Проходим вдоль луча p
    float ray_step = 1.0f; 
    float p_min = -max(width, height) * 0.75f;
    float p_max = max(width, height) * 0.75f;
    
    for (float p = p_min; p <= p_max; p += ray_step) {
        // (t, p) -> (xx, yy)
        float xx = t * c - p * s;
        float yy = -t * s - p * c; // Ускоренный расчет с обратной матрицей

        float x = xx + cx;
        float y = yy + cy;

        // Билинейная интерполяция
        int ix = (int)floorf(x);
        int iy = (int)floorf(y);

        if (ix >= 0 && ix < width - 1 && iy >= 0 && iy < height - 1) {
            float fx = x - ix;
            float fy = y - iy;

            float v00 = slice[iy * width + ix];
            float v10 = slice[iy * width + ix + 1];
            float v01 = slice[(iy + 1) * width + ix];
            float v11 = slice[(iy + 1) * width + ix + 1];

            float v = (1.0f - fx) * (1.0f - fy) * v00 + 
                      fx * (1.0f - fy) * v10 + 
                      (1.0f - fx) * fy * v01 + 
                      fx * fy * v11;
            
            sum += v * ray_step;
        }
    }

    sino[bin_idx * num_angles + angle_idx] = sum;
}

__global__ void multiplyFilterKernel(
    cufftComplex* spectrum,
    const float* filter,
    int pad_size,
    int num_angles) 
{
    // Всего элементов = (pad_size / 2 + 1) * num_angles
    int num_elements = (pad_size / 2 + 1);
    
    int elem_x = blockIdx.x * blockDim.x + threadIdx.x; // index in pad_size/2+1
    int angle_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (elem_x >= num_elements || angle_idx >= num_angles) return;

    int idx = angle_idx * num_elements + elem_x;
    
    // filter[] is symmetric, size is pad_size
    // for cufft R2C, frequency bins are 0 to pad_size/2
    float f = filter[elem_x]; 
    
    spectrum[idx].x *= f;
    spectrum[idx].y *= f;
}

__global__ void backprojectionTextureKernel(
    cudaTextureObject_t tex,
    float* __restrict__ recon,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    int output_size,
    int num_angles,
    int detector_bins,
    float radius,
    float detector_center)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= output_size || y >= output_size) return;

    float xx = (float)x - radius;
    float yy = (float)y - radius;

    if (xx * xx + yy * yy > radius * radius) {
        recon[y * output_size + x] = 0.0f;
        return;
    }

    float sum = 0.0f;
    for (int a = 0; a < num_angles; ++a) {
        float t = xx * cos_table[a] - yy * sin_table[a];
        float u = t + detector_center;
        
        // В Texture Object интерполяция нормализована или ненормализована в зависимости от настроек.
        // Используем ненормализованные координаты: +0.5f для центра пикселя
        sum += tex2D<float>(tex, (float)a + 0.5f, u + 0.5f);
    }

    float scale = (M_PI) / (2.0f * num_angles);
    recon[y * output_size + x] = sum * scale;
}

struct VolToPoint {
    const float* vol_data;
    int w, h, d;
    const float* x_coords;
    const float* y_coords;
    const float* z_coords;

    __host__ __device__
    ct::Point operator()(int idx) const {
        int x = idx % w;
        int y = (idx / w) % h;
        int z = idx / (w * h);
        ct::Point p;
        p.x = x_coords ? x_coords[x] : (float)x;
        p.y = y_coords ? y_coords[y] : (float)y;
        p.z = z_coords ? z_coords[z] : (float)z;
        p.hu = vol_data[idx];
        return p;
    }
};

struct ThresholdPredicate {
    float threshold;
    __host__ __device__ bool operator()(const ct::Point& p) const {
        return p.hu > threshold;
    }
};

} // anonymous namespace

bool CUDABackend::isAvailable() const {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return err == cudaSuccess && count > 0;
}

Sinogram CUDABackend::computeSinogram(const Buffer2D& slice, size_t num_angles, size_t detector_bins) {
    Sinogram sino;
    if (slice.empty() || num_angles == 0 || detector_bins == 0) return sino;

    size_t w = slice.width;
    size_t h = slice.height;

    float cx = (float)(w - 1) * 0.5f;
    float cy = (float)(h - 1) * 0.5f;
    float detector_center = (float)detector_bins / 2.0f;

    sino.data.assign(num_angles, detector_bins, 0.0f);
    sino.angles_deg.resize(num_angles, 0.0f);
    float angle_step = 180.0f / (float)num_angles;
    for (size_t a = 0; a < num_angles; ++a) {
        sino.angles_deg[a] = angle_step * (float)a;
    }
    sino.detector_spacing_mm = 1.0f;

    float* d_slice = nullptr;
    float* d_sino = nullptr;
    float* d_angles = nullptr;

    CUDA_CHECK(cudaMalloc(&d_slice, w * h * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sino, num_angles * detector_bins * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_angles, num_angles * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_slice, slice.data.data(), w * h * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_angles, sino.angles_deg.data(), num_angles * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_sino, 0, num_angles * detector_bins * sizeof(float)));

    dim3 block(16, 16);
    dim3 grid((num_angles + block.x - 1) / block.x, (detector_bins + block.y - 1) / block.y);

    radonForwardKernel<<<grid, block>>>(d_slice, d_sino, d_angles, w, h, num_angles, detector_bins, cx, cy, detector_center);
    // cudaDeviceSynchronize не нужен здесь — cudaMemcpy DeviceToHost сам является барьером

    // Скопировать обратно (заметьте, что sino.data плоская, num_angles * detector_bins)
    // Но в Buffer2D у нас width = num_angles. Kernel писал d_sino[bin_idx * num_angles + angle_idx]. Это совпадает.
    CUDA_CHECK(cudaMemcpy(sino.data.data.data(), d_sino, num_angles * detector_bins * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_slice);
    cudaFree(d_sino);
    cudaFree(d_angles);

    return sino;
}

Buffer2D CUDABackend::reconstructSlice(const Sinogram& sinogram, size_t output_size, const ReconstructionParams& params) {
    Buffer2D recon;
    if (sinogram.data.empty() || output_size == 0) return recon;

    const size_t num_angles = sinogram.data.width;
    const size_t detector_bins = sinogram.data.height;

    const size_t square_bins = static_cast<size_t>(std::ceil(std::sqrt(2.0) * static_cast<double>(detector_bins)));
    const size_t pad = square_bins - detector_bins;
    const size_t pad_before = (square_bins / 2) - (detector_bins / 2);

    const size_t padding_factor = 8;
    const size_t projection_size_padded = std::max<size_t>(64, utils::nextPowerOfTwo(padding_factor * square_bins));

    // Подготовка фильтра на CPU и перенос на GPU
    std::vector<float> cpu_filter = FilterDesign::createFilter(projection_size_padded, params.filter);
    float* d_filter = nullptr;
    CUDA_CHECK(cudaMalloc(&d_filter, projection_size_padded * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_filter, cpu_filter.data(), projection_size_padded * sizeof(float), cudaMemcpyHostToDevice));

    // Копирование Sino
    float* d_sino_padded = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sino_padded, num_angles * projection_size_padded * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_sino_padded, 0, num_angles * projection_size_padded * sizeof(float)));

    // Мы должны копировать строку за строкой. 
    // Вместо сложного кернела здесь, сделаем на CPU и скопируем.
    std::vector<float> cpu_sino_padded(num_angles * projection_size_padded, 0.0f);
    for (size_t a = 0; a < num_angles; ++a) {
        for (size_t b = 0; b < detector_bins; ++b) {
            cpu_sino_padded[a * projection_size_padded + (b + pad_before)] = sinogram.data[b][a];
        }
    }
    CUDA_CHECK(cudaMemcpy(d_sino_padded, cpu_sino_padded.data(), cpu_sino_padded.size() * sizeof(float), cudaMemcpyHostToDevice));

    // cuFFT 1D Batched (R2C и C2R)
    cufftHandle planR2C, planC2R;
    int n[1] = { (int)projection_size_padded };
    cufftPlanMany(&planR2C, 1, n, NULL, 1, projection_size_padded, NULL, 1, projection_size_padded, CUFFT_R2C, num_angles);
    cufftPlanMany(&planC2R, 1, n, NULL, 1, projection_size_padded, NULL, 1, projection_size_padded, CUFFT_C2R, num_angles);

    int complex_size = projection_size_padded / 2 + 1;
    cufftComplex* d_spectrum = nullptr;
    CUDA_CHECK(cudaMalloc(&d_spectrum, num_angles * complex_size * sizeof(cufftComplex)));

    // Forward FFT
    cufftExecR2C(planR2C, (cufftReal*)d_sino_padded, d_spectrum);

    // Apply Filter
    dim3 block(256, 1);
    dim3 grid((complex_size + block.x - 1) / block.x, num_angles);
    multiplyFilterKernel<<<grid, block>>>(d_spectrum, d_filter, projection_size_padded, num_angles);
    // cudaDeviceSynchronize не нужен — cufftExecC2R автоматически ждёт завершения GPU-операций в потоке

    // Inverse FFT
    cufftExecC2R(planC2R, d_spectrum, (cufftReal*)d_sino_padded);
    
    // Normalization for C2R in cuFFT
    float inv_N = 1.0f / projection_size_padded;
    // ... we don't necessarily have to scale here, backprojection output is arbitrary anyway, but let's do it...
    // just applying scaling in CPU copy.

    // Выделяем filtered_square для текстуры (размер num_angles * square_bins)
    std::vector<float> cpu_filtered(num_angles * square_bins);
    CUDA_CHECK(cudaMemcpy(cpu_sino_padded.data(), d_sino_padded, cpu_sino_padded.size() * sizeof(float), cudaMemcpyDeviceToHost));
    
    for (size_t a = 0; a < num_angles; ++a) {
        for (size_t b = 0; b < square_bins; ++b) {
            cpu_filtered[b * num_angles + a] = cpu_sino_padded[a * projection_size_padded + b] * inv_N;
        }
    }

    // Создаем `cudaTextureObject_t` для `cpu_filtered` (размер width=num_angles, height=square_bins)
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray_t cuArray;
    CUDA_CHECK(cudaMallocArray(&cuArray, &channelDesc, num_angles, square_bins));
    CUDA_CHECK(cudaMemcpy2DToArray(cuArray, 0, 0, cpu_filtered.data(), num_angles * sizeof(float), num_angles * sizeof(float), square_bins, cudaMemcpyHostToDevice));

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0; // Используем пиксельные координаты

    cudaTextureObject_t texObj = 0;
    CUDA_CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

    // Backprojection
    float* d_recon = nullptr;
    CUDA_CHECK(cudaMalloc(&d_recon, output_size * output_size * sizeof(float)));

    std::vector<float> cos_table(num_angles);
    std::vector<float> sin_table(num_angles);
    for (size_t a = 0; a < num_angles; ++a) {
        float th = sinogram.angles_deg[a] * (M_PI / 180.0f);
        cos_table[a] = std::cos(th);
        sin_table[a] = std::sin(th);
    }

    float* d_cos = nullptr;
    float* d_sin = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cos, num_angles * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sin, num_angles * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_cos, cos_table.data(), num_angles * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sin, sin_table.data(), num_angles * sizeof(float), cudaMemcpyHostToDevice));

    dim3 bp_block(16, 16);
    dim3 bp_grid((output_size + bp_block.x - 1) / bp_block.x, (output_size + bp_block.y - 1) / bp_block.y);

    backprojectionTextureKernel<<<bp_grid, bp_block>>>(
        texObj, d_recon, d_cos, d_sin, 
        output_size, num_angles, detector_bins, 
        (float)output_size / 2.0f, (float)square_bins / 2.0f
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    recon.assign(output_size, output_size, 0.0f);
    CUDA_CHECK(cudaMemcpy(recon.data.data(), d_recon, output_size * output_size * sizeof(float), cudaMemcpyDeviceToHost));

    // Clean up
    CUDA_CHECK(cudaDestroyTextureObject(texObj));
    CUDA_CHECK(cudaFreeArray(cuArray));
    cudaFree(d_recon);
    cudaFree(d_cos);
    cudaFree(d_sin);
    cudaFree(d_filter);
    cudaFree(d_sino_padded);
    cudaFree(d_spectrum);
    cufftDestroy(planR2C);
    cufftDestroy(planC2R);

    return recon;
}

void CUDABackend::reconstructVolume(const Volume& input_volume, 
                                      Volume& out_reconstruction,
                                      const ReconstructionParams& params,
                                      std::function<void(int slice_idx, const Buffer2D& recon_slice)> onSliceDone) {
    if (input_volume.empty()) return;
    
    // В CUDA бэкенде мы можем итерировать по срезам как на CPU для простоты,
    // либо сделать пакетную обработку. Для начала оставим итерацию + отправку каждого среза.
    const size_t depth = input_volume.depth;
    const size_t width = input_volume.width;
    const size_t height = input_volume.height;
    
    out_reconstruction.assign(width, height, depth, 0.0f);
    out_reconstruction.x_coords = input_volume.x_coords;
    out_reconstruction.y_coords = input_volume.y_coords;
    out_reconstruction.z_coords = input_volume.z_coords;

    for (size_t z = 0; z < depth; ++z) {
        Buffer2D original_slice = input_volume.getSlice(z);
        Sinogram sino = computeSinogram(original_slice, params.num_angles, width);
        Buffer2D recon = reconstructSlice(sino, width, params);
        out_reconstruction.setSlice(z, recon);
        
        if (onSliceDone) {
            onSliceDone(static_cast<int>(z), recon);
        }
    }
}

PointCloud CUDABackend::extractPointCloud(const Volume& vol, float threshold) {
    PointCloud cloud;
    if (vol.empty()) return cloud;

    const size_t N = vol.data.size();
    
    // Перенос данных на устройство
    thrust::device_vector<float> d_vol_data = vol.data;
    thrust::device_vector<float> d_x_coords = vol.x_coords;
    thrust::device_vector<float> d_y_coords = vol.y_coords;
    thrust::device_vector<float> d_z_coords = vol.z_coords;

    VolToPoint to_point_op;
    to_point_op.vol_data = thrust::raw_pointer_cast(d_vol_data.data());
    to_point_op.w = vol.width;
    to_point_op.h = vol.height;
    to_point_op.d = vol.depth;
    to_point_op.x_coords = vol.x_coords.empty() ? nullptr : thrust::raw_pointer_cast(d_x_coords.data());
    to_point_op.y_coords = vol.y_coords.empty() ? nullptr : thrust::raw_pointer_cast(d_y_coords.data());
    to_point_op.z_coords = vol.z_coords.empty() ? nullptr : thrust::raw_pointer_cast(d_z_coords.data());

    thrust::counting_iterator<int> iter(0);
    auto transform_iter = thrust::make_transform_iterator(iter, to_point_op);

    ThresholdPredicate pred;
    pred.threshold = threshold;

    int count = thrust::count_if(transform_iter, transform_iter + N, pred);
    if (count <= 0) return cloud;

    thrust::device_vector<ct::Point> d_cloud(count);
    thrust::copy_if(transform_iter, transform_iter + N, d_cloud.begin(), pred);

    cloud.resize(count);
    thrust::copy(d_cloud.begin(), d_cloud.end(), cloud.begin());

    return cloud;
}

} // namespace ct
