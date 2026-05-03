#pragma once

// FFTBackendBuiltin.h — реализация IFFTBackend через учебный Cooley–Tukey (FFT.cpp).
//
// Используется по умолчанию (CT_FFT_BACKEND_FFTW не определён) и всегда
// для SerialBackend (чтобы не вводить зависимость от FFTW в reference-коде).
//
// Pinned-план: объект хранит FFT::Plan fwd_plan_ / inv_plan_, пересозданных
// при первом вызове prepare() или при смене размера n.

#include "IFFTBackend.h"
#include "FFT.h"

namespace ct {

class FFTBackendBuiltin final : public IFFTBackend {
public:
    void prepare(size_t n) override;

    void forwardReal(const std::vector<float>& input,
                     std::vector<Complex>& output) override;

    void inverseReal(const std::vector<Complex>& input,
                     std::vector<float>& output) override;

    const char* name() const noexcept override { return "Builtin (Cooley-Tukey)"; }

private:
    size_t       n_     = 0;
    FFT::Plan    fwd_;        // план прямого БПФ
    FFT::Plan    inv_;        // план обратного БПФ
};

} // namespace ct
