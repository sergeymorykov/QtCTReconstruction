#pragma once

#include "CTTypes.h"
#include <array>
#include <cstdint>
#include <memory>

namespace ct {

class IVolumeGenerator;

class Generator3D {
public:
    enum class BackendType { CPU, CUDA };

    struct TissueParams {
        float hu_min = -20.0f;
        float hu_max = 45.0f;
        float size_min_mm = 3.0f;
        float size_max_mm = 10.0f;
        bool peripheral_only = false;
        float peripheral_radius_min = 0.0f; // fraction of brain_radius_mm
        float peripheral_radius_max = 0.0f; // fraction of brain_radius_mm
        float probability = 1.0f;
    };

    struct Params {
        std::array<size_t, 3> shape = {128, 128, 64}; // [x, y, z]
        float brain_radius_mm = 50.0f;
        size_t num_ellipsoids = 120;
        uint32_t seed = 42;
        TissueParams soft_tissue{};
        TissueParams bone_tissue{200.0f, 1000.0f, 2.0f, 8.0f, true, 0.7f, 0.9f, 0.15f};
    };

    Generator3D();
    ~Generator3D();

    void setBackend(BackendType type);
    Volume generateBrainHU(const Params& params);

    // Статический метод для обратной совместимости — просто CPU-генерация
    static Volume generate(const Params& params);

private:
    std::unique_ptr<IVolumeGenerator> m_impl;
};

#ifdef __CUDACC__
#define CT_HD __host__ __device__
#else
#define CT_HD
#endif

inline CT_HD float calculateDistance(float x, float y, float z) {
    return x * x + y * y + z * z;
}

inline CT_HD bool isInsideEllipsoid(float dx, float dy, float dz, float ax, float ay, float az) {
    const float norm = (dx * dx) / (ax * ax) + (dy * dy) / (ay * ay) + (dz * dz) / (az * az);
    return norm <= 1.0f;
}

} // namespace ct
