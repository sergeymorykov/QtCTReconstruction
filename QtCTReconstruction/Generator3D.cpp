#include "Generator3D.h"
#include "Generator3DCPU.h"
#include "Generator3DGPU.h"
#include "Utils.h"
#include <iostream>

namespace ct {

Generator3D::Generator3D() {
    // Default to CPU
    m_impl = std::make_unique<Generator3DCPU>();
}

Generator3D::~Generator3D() = default;

void Generator3D::setBackend(BackendType type) {
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
}

Volume Generator3D::generateBrainHU(const Params& params) {
    if (!m_impl) return Volume();
    return m_impl->generateBrainHU(params);
}

Volume Generator3D::generate(const Params& params) {
    // Helper static method for one-off generation
    Generator3D gen;
    // Try GPU first by default if static generate is called? 
    // Or stick to CPU for safety? The user's request suggests 
    // they want to be able to switch, but for "generate(params)" let's try GPU if available.
    auto gpu = std::make_unique<Generator3DGPU>();
    if (gpu->isAvailable()) {
        return gpu->generateBrainHU(params);
    }
    return Generator3DCPU().generateBrainHU(params);
}

} // namespace ct
