#pragma once

// FFTBackendFFTW.h — реализация IFFTBackend через FFTW3 (libfftw3f).
//
// Подключается при -DFFT_BACKEND=FFTW (CMake добавляет CT_FFT_BACKEND_FFTW).
// Ожидание: ×3–5 по сравнению с учебным Cooley–Tukey на стадии фильтрации.
//
// Pinned-планы:
//   fftwf_plan_dft_r2c_1d  — один экземпляр на поток, размер n, FFTW_MEASURE
//   fftwf_plan_dft_c2r_1d  — один экземпляр на поток, размер n, FFTW_MEASURE
//   Планы создаются в prepare() при первом вызове или смене размера.
//   Буферы in_/out_ — плоские aligned-аллокации fftwf_alloc_*.
//
// FFTW thread-safety: каждый поток имеет собственный объект (через threadLocal()),
// поэтому glобальный мьютекс нужен только при fftwf_plan_* (создание плана
// требует синхронизации внутри FFTW при первом вызове).

#ifdef CT_FFT_BACKEND_FFTW
#include "IFFTBackend.h"
#include <fftw3.h>

namespace ct {

class FFTBackendFFTW final : public IFFTBackend {
public:
    FFTBackendFFTW() = default;
    ~FFTBackendFFTW() override;

    // Создать/пересоздать планы для нового размера n (в вещественных отсчётах).
    void prepare(size_t n) override;

    // R→C: n вещественных → n/2+1 комплексных
    void forwardReal(const std::vector<float>& input,
                     std::vector<Complex>& output) override;

    // C→R: n/2+1 комплексных → n вещественных
    void inverseReal(const std::vector<Complex>& input,
                     std::vector<float>& output) override;

    const char* name() const noexcept override { return "FFTW3 (single precision)"; }

private:
    void destroyPlans();

    size_t         n_    = 0;
    fftwf_plan     pfwd_ = nullptr;   // plan_dft_r2c_1d
    fftwf_plan     pinv_ = nullptr;   // plan_dft_c2r_1d
    float*         in_   = nullptr;   // fftwf_alloc_real(n_)
    fftwf_complex* out_  = nullptr;   // fftwf_alloc_complex(n_/2+1)
};

} // namespace ct
#endif // CT_FFT_BACKEND_FFTW
