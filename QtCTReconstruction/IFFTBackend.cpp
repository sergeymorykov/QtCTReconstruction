// IFFTBackend.cpp — фабрика threadLocal() для IFFTBackend.
//
// threadLocal() возвращает thread_local экземпляр нужного бэкенда.
// Выбор бэкенда определяется препроцессорным макросом:
//   CT_FFT_BACKEND_FFTW → FFTBackendFFTW  (FFTW3, -DFFT_BACKEND=FFTW)
//   иначе               → FFTBackendBuiltin (учебный Cooley–Tukey, по умолчанию)
//
// Каждый OpenMP-поток имеет собственный объект:
//   - нет гонок за состояние плана
//   - нет блокировок при forwardReal/inverseReal
//   - планы создаются один раз на поток (при первом вызове prepare())

#include "IFFTBackend.h"

#ifdef CT_FFT_BACKEND_FFTW
#include "FFTBackendFFTW.h"
#else
#include "FFTBackendBuiltin.h"
#endif

namespace ct {

IFFTBackend& IFFTBackend::threadLocal() {
#ifdef CT_FFT_BACKEND_FFTW
    // FFTW3: один план на поток, создаётся при первом вызове prepare()
    thread_local FFTBackendFFTW instance;
#else
    // Builtin Cooley–Tukey: один объект на поток
    thread_local FFTBackendBuiltin instance;
#endif
    return instance;
}

} // namespace ct
