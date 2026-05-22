# QtCTReconstruction

Десктоп-приложение на Qt 6 / C++17 для реконструкции 3D-объёмов компьютерной томографии методом фильтрованной обратной проекции (FBP) на основе преобразования Радона.

## Возможности

- Генерация синтетических 3D-фантомов (CPU и GPU)
- Прямое преобразование Радона (получение синограмм/проекций)
- Реконструкция методом FBP с настраиваемыми частотными фильтрами (Ram-Lak, Shepp-Logan, Hann и др.)
- Несколько вычислительных бэкендов с горячим переключением:
  - **Serial** — эталонный однопоточный (для регрессий, Debug)
  - **OpenMP** — многопоточный CPU
  - **CUDA** — GPU-ускорение
  - **Hybrid** — гибрид CPU+GPU
- Сменные FFT-бэкенды: встроенный Cooley–Tukey или FFTW3 (single precision)
- Qt Quick 3D UI: просмотр срезов и облака точек объёма

## Зависимости

- CMake ≥ 3.18
- Компилятор с C++17 (MSVC рекомендуется на Windows)
- **Qt 6** с модулями: Core, Gui, Widgets, Quick, QuickControls2, QuickLayouts, 3DCore, 3DRender, 3DExtras, 3DQuick, 3DQuickScene3D
- **CUDA Toolkit** (архитектуры 75/80/86/89 — Turing/Ampere/Ada)
- **OpenMP**
- Опционально: **FFTW3** (libfftw3f) для ускоренной FFT

## Сборка

Задайте путь к Qt через переменную окружения `QTDIR`, затем:

```powershell
# Конфигурация (Release)
cmake -S . -B out/build -DCMAKE_BUILD_TYPE=Release

# Сборка
cmake --build out/build --config Release
```

### Опции CMake

| Опция | Значения | По умолчанию | Описание |
|---|---|---|---|
| `FFT_BACKEND` | `Builtin` / `FFTW` | `Builtin` | Реализация FFT |
| `BUILD_SERIAL_BACKEND` | `ON` / `OFF` | `ON` в Debug, `OFF` в Release | Эталонный однопоточный бэкенд |
| `CT_CUDA_FAST_DEBUG` | `ON` / `OFF` | `OFF` | Убрать `-G` из Debug-сборки CUDA (ядра компилируются с оптимизацией) |

Пример со всеми опциями:

```powershell
cmake -S . -B out/build -DCMAKE_BUILD_TYPE=Release -DFFT_BACKEND=FFTW -DBUILD_SERIAL_BACKEND=OFF
```

## Запуск

После сборки на Windows автоматически вызывается `windeployqt`, который копирует нужные Qt-DLL рядом с исполняемым файлом.

```powershell
./out/build/Release/QtCTReconstruction.exe
```

## Структура

- `QtCTReconstruction/` — исходники C++/CUDA/QML
- `*.cu` — CUDA-ядра (`CUDABackend`, `HybridBackend`, `Generator3DGPU`)
- `*.qml` — UI (`main.qml`, `PointCloudWindow.qml`, `ImageWithAxes.qml`)
- `_diary_work/`, `optim.md` — рабочие заметки по оптимизации
