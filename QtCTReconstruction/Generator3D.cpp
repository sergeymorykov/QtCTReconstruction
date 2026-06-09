#include "Generator3D.h"
#include "Generator3DCPU.h"
#ifndef CT_NO_CUDA
#include "Generator3DGPU.h"
#endif
#include "Utils.h"
#include <iostream>

namespace ct {

Generator3D::Generator3D() {
    // Default to CPU
    m_impl = std::make_unique<Generator3DCPU>();
}

Generator3D::~Generator3D() = default;

void Generator3D::setBackend(BackendType type) {
#ifndef CT_NO_CUDA
    if (type == BackendType::CUDA) {
        auto gpu = std::make_unique<Generator3DGPU>();
        if (gpu->isAvailable()) {
            m_impl = std::move(gpu);
        } else {
            std::cerr << "CUDA is not available. Falling back to CPU." << std::endl;
            m_impl = std::make_unique<Generator3DCPU>();
        }
    } else {
        m_impl = std::make_unique<Generator3DCPU>();
    }
#else
    // Сборка без CUDA: всегда CPU-генератор, независимо от запрошенного типа.
    if (type == BackendType::CUDA) {
        std::cerr << "Build without CUDA: GPU generator unavailable, using CPU." << std::endl;
    }
    (void)type;
    m_impl = std::make_unique<Generator3DCPU>();
#endif
}

Volume Generator3D::generateBrainHU(const Params& params) {
    if (!m_impl) return Volume();
    return m_impl->generateBrainHU(params);
}

Volume Generator3D::generate(const Params& params) {
    // Helper static method for one-off generation
    Generator3D gen;
#ifndef CT_NO_CUDA
    // Try GPU first if available, otherwise fall back to CPU.
    auto gpu = std::make_unique<Generator3DGPU>();
    if (gpu->isAvailable()) {
        return gpu->generateBrainHU(params);
    }
#endif
    return Generator3DCPU().generateBrainHU(params);
}

} // namespace ct
