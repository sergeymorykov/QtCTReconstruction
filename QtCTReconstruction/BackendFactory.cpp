#include "IReconstructionBackend.h"
#ifdef CT_HAS_SERIAL_BACKEND
#include "SerialBackend.h"
#endif
#include "OpenMPBackend.h"
#include "CUDABackend.h"
#include "HybridBackend.h"

namespace ct {

std::shared_ptr<IReconstructionBackend> BackendFactory::create(BackendType type) {
#ifdef CT_HAS_SERIAL_BACKEND
    if (type == BackendType::Serial) {
        return std::make_shared<SerialBackend>();
    } else
#endif
    if (type == BackendType::OpenMP) {
        return std::make_shared<OpenMPBackend>();
    } else if (type == BackendType::CUDA) {
        return std::make_shared<CUDABackend>();
    } else if (type == BackendType::Hybrid) {
        return std::make_shared<HybridBackend>();
    }
    return nullptr;
}

std::shared_ptr<IReconstructionBackend> BackendFactory::createBestAvailable() {
    auto cudaBackend = create(BackendType::CUDA);
    if (cudaBackend && cudaBackend->isAvailable()) {
        return cudaBackend;
    }
    return create(BackendType::OpenMP);
}

} // namespace ct
