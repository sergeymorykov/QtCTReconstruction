// FFTBackendFFTW.cpp — реализация IFFTBackend через FFTW3 (single precision).
//
// Скомпилируется только при -DFFT_BACKEND=FFTW (CMake определяет CT_FFT_BACKEND_FFTW).
//
// Ключевые решения:
//   • fftwf_plan создаётся с FFTW_MEASURE в prepare() — FFTW перебирает алгоритмы
//     и выбирает оптимальный для данного CPU. Одноразовая стоимость: ~50–300 мс.
//   • Буферы in_/out_ выделяются fftwf_alloc_* (16-байт SIMD-выравнивание).
//   • Глобальный std::mutex guards_fftw_plan_creation: fftwf_plan_* не потокобезопасен
//     при первом обращении к FFTW; последующие execute() потокобезопасны.
//   • Нормировка IFFT: FFTW не нормирует — делим на n вручную.

#ifdef CT_FFT_BACKEND_FFTW

#include "FFTBackendFFTW.h"

#include <mutex>
#include <stdexcept>
#include <cstring>   // memcpy

namespace ct {

// FFTW3 требует внешней синхронизации только при создании/удалении планов.
// Параллельное execute() безопасно без блокировки.
static std::mutex g_fftw_plan_mutex;

// ----------------------------------------------------------------
// Деструктор: освобождаем план и буферы
// ----------------------------------------------------------------
FFTBackendFFTW::~FFTBackendFFTW() {
    destroyPlans();
}

void FFTBackendFFTW::destroyPlans() {
    std::lock_guard<std::mutex> lock(g_fftw_plan_mutex);
    if (pfwd_) { fftwf_destroy_plan(pfwd_); pfwd_ = nullptr; }
    if (pinv_) { fftwf_destroy_plan(pinv_); pinv_ = nullptr; }
    if (in_)   { fftwf_free(in_);   in_  = nullptr; }
    if (out_)  { fftwf_free(out_);  out_ = nullptr; }
    n_ = 0;
}

// ----------------------------------------------------------------
// prepare(): создать/пересоздать планы для размера n
// ----------------------------------------------------------------
void FFTBackendFFTW::prepare(size_t n) {
    if (n_ == n) return;   // план актуален

    if (n == 0 || (n & (n - 1)) != 0)
        throw std::invalid_argument("FFTBackendFFTW: n must be a power of two");

    // Удаляем старые планы
    destroyPlans();

    // Выделяем aligned-буферы
    in_  = fftwf_alloc_real(n);
    out_ = fftwf_alloc_complex(n / 2 + 1);
    if (!in_ || !out_)
        throw std::runtime_error("FFTBackendFFTW: fftwf_alloc failed");

    // Создание планов — потребует внешней блокировки
    {
        std::lock_guard<std::mutex> lock(g_fftw_plan_mutex);
        // FFTW_MEASURE: ~50–300 мс однократно; даёт оптимальный код для этого CPU
        pfwd_ = fftwf_plan_dft_r2c_1d(
            static_cast<int>(n), in_, out_, FFTW_MEASURE | FFTW_DESTROY_INPUT);
        pinv_ = fftwf_plan_dft_c2r_1d(
            static_cast<int>(n), out_, in_, FFTW_MEASURE | FFTW_DESTROY_INPUT);
    }

    if (!pfwd_ || !pinv_)
        throw std::runtime_error("FFTBackendFFTW: fftwf_plan failed");

    n_ = n;
}

// ----------------------------------------------------------------
// forwardReal(): R→C, n вещественных → n/2+1 комплексных
// ----------------------------------------------------------------
void FFTBackendFFTW::forwardReal(const std::vector<float>& input,
                                  std::vector<Complex>& output) {
    if (input.size() != n_)
        throw std::invalid_argument("FFTBackendFFTW::forwardReal: input size mismatch");

    // Копируем во входной буфер FFTW
    std::memcpy(in_, input.data(), n_ * sizeof(float));

    // Выполняем план (потокобезопасно)
    fftwf_execute(pfwd_);

    // Копируем результат в std::complex<float>
    const size_t nout = n_ / 2 + 1;
    output.resize(nout);
    for (size_t k = 0; k < nout; ++k) {
        output[k] = Complex(out_[k][0], out_[k][1]);
    }
}

// ----------------------------------------------------------------
// inverseReal(): C→R, n/2+1 комплексных → n вещественных (не нормировано!)
// FFTW возвращает сумму без деления на n — нормируем вручную.
// ----------------------------------------------------------------
void FFTBackendFFTW::inverseReal(const std::vector<Complex>& input,
                                  std::vector<float>& output) {
    const size_t nout = n_ / 2 + 1;
    if (input.size() != nout)
        throw std::invalid_argument("FFTBackendFFTW::inverseReal: input size mismatch");

    // Копируем во входной буфер FFTW (out_ — комплексный вход для C2R)
    for (size_t k = 0; k < nout; ++k) {
        out_[k][0] = input[k].real();
        out_[k][1] = input[k].imag();
    }

    // Выполняем план
    fftwf_execute(pinv_);

    // Копируем и нормируем: FFTW c2r не делит на n
    output.resize(n_);
    const float inv_n = 1.0f / static_cast<float>(n_);
    for (size_t i = 0; i < n_; ++i) {
        output[i] = in_[i] * inv_n;
    }
}

} // namespace ct

#endif // CT_FFT_BACKEND_FFTW
