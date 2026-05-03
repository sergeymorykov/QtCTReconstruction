#pragma once

// IFFTBackend.h — абстракция над реализацией БПФ.
//
// Цель: позволить переключать FFT-движок через CMake без изменения вызывающего кода:
//   -DFFT_BACKEND=Builtin  (по умолчанию) — учебный Cooley–Tukey из FFT.cpp
//   -DFFT_BACKEND=FFTW     — FFTW3 с pinned-планами (×3–5 ускорение на фильтрации)
//
// Использование:
//   auto& fft = IFFTBackend::threadLocal();   // один объект на поток
//   fft.prepare(fft_size);                    // один раз на размер
//   fft.forwardReal(proj_padded, spectrum);
//   fft.inverseReal(spectrum, q);
//
// Thread-safety: каждый поток владеет собственным экземпляром через threadLocal(),
// поэтому никакой блокировки не требуется.

#include "CTTypes.h"
#include <vector>
#include <memory>

namespace ct {

class IFFTBackend {
public:
    virtual ~IFFTBackend() = default;

    // Подготовить план для заданного размера (вызывается один раз или при смене размера).
    // Реализации кешируют план и не пересоздают его, если n не изменился.
    virtual void prepare(size_t n) = 0;

    // Real-to-Complex FFT. Размер input должен равняться n, переданному в prepare().
    // output будет иметь n/2+1 элементов.
    virtual void forwardReal(const std::vector<float>& input,
                             std::vector<Complex>& output) = 0;

    // Complex-to-Real IFFT. Размер input должен равняться n/2+1.
    // output будет иметь n элементов.
    virtual void inverseReal(const std::vector<Complex>& input,
                             std::vector<float>& output) = 0;

    // ----------------------------------------------------------------
    // Фабрика: создаёт объект нужного типа в зависимости от
    // макроса CT_FFT_BACKEND_FFTW (определяется через CMake).
    // Каждый поток хранит свой экземпляр — никаких блокировок.
    // ----------------------------------------------------------------
    static IFFTBackend& threadLocal();

    // Имя бэкенда (для логирования/UI)
    virtual const char* name() const noexcept = 0;
};

} // namespace ct
