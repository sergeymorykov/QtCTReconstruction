#include "IReconstructionBackend.h"
#include "OpenMPBackend.h"
#include "CUDABackend.h"
#include "CUDAPureBackend.h"

namespace ct {

std::shared_ptr<IReconstructionBackend> BackendFactory::create(BackendType type) {
    if (type == BackendType::OpenMP) {
        return std::make_shared<OpenMPBackend>();
    } else if (type == BackendType::CUDA) {
        return std::make_shared<CUDABackend>();
    } else if (type == BackendType::CUDAPure) {
        return std::make_shared<CUDAPureBackend>();
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
