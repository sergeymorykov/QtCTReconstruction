cmake_minimum_required(VERSION 3.18)
project(QtCTReconstruction LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)

# Qt 6
find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets Quick QuickControls2 QuickLayouts 3DCore 3DRender 3DExtras 3DQuick 3DQuickScene3D)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

# OpenMP (required for fallback & CPU ops)
find_package(OpenMP REQUIRED)

# CUDA setup
find_package(CUDAToolkit REQUIRED)

# ============================================================
# Producer / Consumer split
# ------------------------------------------------------------
# Из одного source tree собираем ДВА exe:
#   * QtCTReconstruction          — Producer (Generate → Reconstruct → Export Sino)
#   * QtCTReconstructionConsumer  — Consumer (Load Sinograms → Reconstruct → Export PNG)
# Различия только в main.{cpp,qml} и compile-time дефайнах CT_PRODUCER_BUILD /
# CT_CONSUMER_BUILD. Всё остальное (backend'ы, controller, FileIO, ...) общее.
# ============================================================
set(CT_COMMON_SOURCES
    QtCTReconstruction/AppWindow.cpp
    QtCTReconstruction/CtReconstructionController.cpp
    QtCTReconstruction/CtSliceImageProvider.cpp
    QtCTReconstruction/FileIO.cpp
    QtCTReconstruction/Utils.cpp
    QtCTReconstruction/Generator3D.cpp
    QtCTReconstruction/Generator3DCPU.cpp
    QtCTReconstruction/Generator3DGPU.cu
    QtCTReconstruction/RadonTransform.cpp
    QtCTReconstruction/FilterDesign.cpp
    QtCTReconstruction/FilteredBackprojection.cpp
    QtCTReconstruction/FFT.cpp
    QtCTReconstruction/BackendFactory.cpp
    QtCTReconstruction/OpenMPBackend.cpp
    QtCTReconstruction/CUDABackend.cu
    QtCTReconstruction/HybridBackend.cu
    QtCTReconstruction/PointCloudGeometry.cpp
    QtCTReconstruction/OptimizedBackprojectionCPU.cpp
    QtCTReconstruction/ProjectionGeometry.cpp
    # FFT абстракция (Sprint 1)
    QtCTReconstruction/IFFTBackend.cpp
    QtCTReconstruction/FFTBackendBuiltin.cpp
)

# ============================================================
# FFT бэкенд (переключается через cmake -DFFT_BACKEND=Builtin|FFTW)
# Builtin: учебный Cooley–Tukey (без внешних зависимостей)
# FFTW:    FFTW3 single-precision (требует libfftw3f, ожидание x3-5)
# ============================================================
set(FFT_BACKEND "Builtin" CACHE STRING "FFT implementation: Builtin or FFTW")
set_property(CACHE FFT_BACKEND PROPERTY STRINGS Builtin FFTW)

if(FFT_BACKEND STREQUAL "FFTW")
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(FFTW3F fftw3f)
    endif()
    if(NOT FFTW3F_FOUND)
        # Ручной поиск через find_library
        find_library(FFTW3F_LIB NAMES fftw3f fftw3f-3 libfftw3f-3
            HINTS
                "$ENV{FFTW_DIR}/lib"
                "C:/fftw"
                "C:/Program Files/fftw"
        )
        find_path(FFTW3F_INCLUDE_DIR NAMES fftw3.h
            HINTS
                "$ENV{FFTW_DIR}/include"
                "C:/fftw"
                "C:/Program Files/fftw"
        )
        if(FFTW3F_LIB AND FFTW3F_INCLUDE_DIR)
            set(FFTW3F_FOUND TRUE)
            set(FFTW3F_LIBRARIES ${FFTW3F_LIB})
            set(FFTW3F_INCLUDE_DIRS ${FFTW3F_INCLUDE_DIR})
        endif()
    endif()

    if(FFTW3F_FOUND)
        message(STATUS "[CT] FFT backend: FFTW3 (single precision)")
        message(STATUS "[CT] FFTW3 include: ${FFTW3F_INCLUDE_DIRS}")
        message(STATUS "[CT] FFTW3 lib:     ${FFTW3F_LIBRARIES}")
    else()
        message(WARNING "[CT] FFT_BACKEND=FFTW but fftw3f not found; falling back to Builtin")
        set(FFT_BACKEND "Builtin" CACHE STRING "FFT implementation" FORCE)
    endif()
else()
    message(STATUS "[CT] FFT backend: Builtin (Cooley-Tukey)")
endif()

# ============================================================
# Serial backend (эталонная однопоточная реализация)
# По умолчанию включён в Debug (для регрессионных тестов),
# выключен в Release (не нужен в продакшн-сборке).
# Переопределить: cmake -DBUILD_SERIAL_BACKEND=ON/OFF
# ============================================================
if(NOT DEFINED BUILD_SERIAL_BACKEND)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR NOT CMAKE_BUILD_TYPE)
        set(BUILD_SERIAL_BACKEND ON CACHE BOOL "Build serial reference backend" FORCE)
    else()
        set(BUILD_SERIAL_BACKEND OFF CACHE BOOL "Build serial reference backend" FORCE)
    endif()
endif()

# ============================================================
# Иконка приложения
# ------------------------------------------------------------
# Сгенерируйте один раз: python resources/generate_icon.py
# (требует pip install cairosvg pillow). После генерации re-configure
# НЕ нужен — POST_BUILD copy для PNG и сам .rc embed для .ico
# подхватятся автоматически: copy_if_different + проверка через CMake-скрипт.
# ============================================================
set(CT_APP_ICON_RC  "${CMAKE_CURRENT_SOURCE_DIR}/resources/app_icon.rc")
set(CT_APP_ICON_ICO "${CMAKE_CURRENT_SOURCE_DIR}/resources/icon.ico")
set(CT_APP_ICON_PNG "${CMAKE_CURRENT_SOURCE_DIR}/resources/icon.png")
# .rc-файл всегда добавляем в исходники — RC-компилятор сам проверит ICO
# и завершится с ошибкой если файла нет. Чтобы избежать ошибки сборки в
# отсутствии иконки, .rc-файл условно подключается через EXISTS только для
# Windows. Если ICO нет на момент configure — печатаем предупреждение.
if(WIN32 AND EXISTS "${CT_APP_ICON_ICO}")
    set(_ct_icon_sources "${CT_APP_ICON_RC}")
    message(STATUS "[CT] App icon (.ico embedded into .exe): ${CT_APP_ICON_ICO}")
else()
    set(_ct_icon_sources "")
    if(WIN32)
        message(STATUS "[CT] App icon NOT found at ${CT_APP_ICON_ICO}")
        message(STATUS "     Run: python resources/generate_icon.py")
        message(STATUS "     Затем перезапустите cmake configure для embed в .exe")
    endif()
endif()

# ============================================================
# Два exe: Producer (основной) + Consumer (для FBP по готовым синограммам)
# ============================================================
qt_add_executable(QtCTReconstruction MANUAL_FINALIZATION
    ${CT_COMMON_SOURCES}
    QtCTReconstruction/main.cpp
    ${_ct_icon_sources}
)
target_compile_definitions(QtCTReconstruction PRIVATE CT_PRODUCER_BUILD)

qt_add_executable(QtCTReconstructionConsumer MANUAL_FINALIZATION
    ${CT_COMMON_SOURCES}
    QtCTReconstruction/main_consumer.cpp
    ${_ct_icon_sources}
)
target_compile_definitions(QtCTReconstructionConsumer PRIVATE CT_CONSUMER_BUILD)

# Консольное окно:
#   * Debug         → CONSOLE subsystem (qDebug() пишет в stdout, удобно для разработки)
#   * Release / RWDI / MinSizeRel → WINDOWS subsystem (консоль НЕ открывается при
#     запуске из Explorer; qDebug по-прежнему виден через OutputDebugString
#     в DebugView / VS Output).
# WIN32_EXECUTABLE поддерживает генератор-выражения (CMake ≥ 3.0); используем
# $<NOT:$<CONFIG:Debug>> чтобы получить per-config поведение.
set_target_properties(QtCTReconstruction QtCTReconstructionConsumer PROPERTIES
    WIN32_EXECUTABLE $<NOT:$<CONFIG:Debug>>
)

# ============================================================
# Иконка в Qt-resource (для setWindowIcon в рантайме).
# QIcon(":/icons/icon.png") работает гарантированно если PNG был на месте
# в момент cmake configure. QT_RESOURCE_ALIAS устанавливает фиксированное
# имя в ресурс-системе (иначе абсолютный путь из FILES может попасть в URI).
# ============================================================
if(EXISTS "${CT_APP_ICON_PNG}")
    set_source_files_properties("${CT_APP_ICON_PNG}" PROPERTIES
        QT_RESOURCE_ALIAS "icon.png"
    )
    foreach(_tgt IN ITEMS QtCTReconstruction QtCTReconstructionConsumer)
        qt_add_resources(${_tgt} "${_tgt}_app_icon"
            PREFIX "/icons"
            FILES "${CT_APP_ICON_PNG}"
        )
    endforeach()
    message(STATUS "[CT] App icon (.png embedded into Qt resource): ${CT_APP_ICON_PNG}")
    message(STATUS "     Resource path: :/icons/icon.png")
else()
    message(STATUS "[CT] App icon PNG not found — runtime icon will be Qt default.")
    message(STATUS "     Run: python resources/generate_icon.py + re-configure cmake")
endif()

# QML модули — у каждого свой URI, чтобы ресурсы не конфликтовали.
# Producer'у нужны 3D-viewer (PointCloudWindow) и осевая обёртка (ImageWithAxes).
# Consumer'у достаточно ImageWithAxes — точечного облака у него нет.
qt_add_qml_module(QtCTReconstruction
    URI QtCTReconstruction
    VERSION 1.0
    QML_FILES
        QtCTReconstruction/main.qml
        QtCTReconstruction/PointCloudWindow.qml
        QtCTReconstruction/ImageWithAxes.qml
)

qt_add_qml_module(QtCTReconstructionConsumer
    URI QtCTReconstructionConsumer
    VERSION 1.0
    QML_FILES
        QtCTReconstruction/main_consumer.qml
        QtCTReconstruction/ImageWithAxes.qml
)

# CUDA optimisation flags (выносим до foreach, т.к. используются внутри него)
set(_ct_cuda_release_flags
    -O3
    --use_fast_math
    -lineinfo
    --ptxas-options=-v
)
if(MSVC)
    list(APPEND _ct_cuda_release_flags -Xcompiler=/O2)
endif()

# ============================================================
# Общая конфигурация ДЛЯ ОБОИХ таргетов
# ============================================================
foreach(_tgt IN ITEMS QtCTReconstruction QtCTReconstructionConsumer)

    # --- FFT backend ---
    if(FFT_BACKEND STREQUAL "FFTW" AND FFTW3F_FOUND)
        target_sources(${_tgt} PRIVATE
            QtCTReconstruction/FFTBackendFFTW.cpp
        )
        target_compile_definitions(${_tgt} PRIVATE CT_FFT_BACKEND_FFTW)
        target_include_directories(${_tgt} PRIVATE ${FFTW3F_INCLUDE_DIRS})
        target_link_libraries(${_tgt} PRIVATE ${FFTW3F_LIBRARIES})
    endif()

    # --- Serial backend ---
    if(BUILD_SERIAL_BACKEND)
        target_sources(${_tgt} PRIVATE QtCTReconstruction/SerialBackend.cpp)
        target_compile_definitions(${_tgt} PRIVATE CT_HAS_SERIAL_BACKEND)
    endif()

    target_include_directories(${_tgt} PRIVATE QtCTReconstruction)

    target_link_libraries(${_tgt} PRIVATE
        Qt6::Core
        Qt6::Gui
        Qt6::Widgets
        Qt6::Quick
        Qt6::QuickControls2
        Qt6::QuickLayouts
        Qt6::3DCore
        Qt6::3DRender
        Qt6::3DExtras
        Qt6::3DQuick
        Qt6::3DQuickScene3D
        OpenMP::OpenMP_CXX
        CUDA::cudart
        CUDA::cufft
        gdiplus
        comctl32
    )

    # --- MSVC: OpenMP + EH ---
    if(MSVC)
        target_compile_options(${_tgt} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/openmp:llvm>
            $<$<COMPILE_LANGUAGE:CXX>:/openmp:experimental>
            $<$<COMPILE_LANGUAGE:CXX>:/EHsc>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/openmp:llvm>
        )
    endif()

    # --- Debug/Release дефайны ---
    target_compile_definitions(${_tgt} PRIVATE
        $<$<CONFIG:Debug>:CT_DEBUG_MODE>
        $<$<CONFIG:Release>:CT_RELEASE_MODE>
    )

    # --- CUDA Release flags ---
    target_compile_options(${_tgt} PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:${_ct_cuda_release_flags}>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:RelWithDebInfo>>:${_ct_cuda_release_flags}>
    )

    # --- Release с PDB-символами ---
    if(MSVC)
        target_compile_options(${_tgt} PRIVATE
            $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/Zi>
            $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-Xcompiler=/Zi>
            $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RelWithDebInfo>>:/Zi>
            $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:RelWithDebInfo>>:-Xcompiler=/Zi>
        )
        target_link_options(${_tgt} PRIVATE
            $<$<CONFIG:Release>:/DEBUG>
            $<$<CONFIG:Release>:/OPT:REF>
            $<$<CONFIG:Release>:/OPT:ICF>
            $<$<CONFIG:RelWithDebInfo>:/DEBUG>
            $<$<CONFIG:RelWithDebInfo>:/OPT:REF>
            $<$<CONFIG:RelWithDebInfo>:/OPT:ICF>
        )
    endif()

    qt_import_qml_plugins(${_tgt})
    qt_finalize_executable(${_tgt})

endforeach()

# --- Логи о Serial backend (один раз, не в foreach) ---
if(BUILD_SERIAL_BACKEND)
    message(STATUS "[CT] SerialBackend: ENABLED (both targets)")
else()
    message(STATUS "[CT] SerialBackend: DISABLED (Release build)")
endif()

# --- Глобальные MSVC флаги (не per-target) ---
if(MSVC)
    add_compile_options(/utf-8 /Zc:preprocessor)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler /Zc:preprocessor")
endif()

# ============================================================
# CUDA Debug режим (фастер дебаг без -G)
# ============================================================
option(CT_CUDA_FAST_DEBUG
       "Strip -G from Debug CUDA flags (kernels run at full speed; host -g kept)"
       OFF)

if(CT_CUDA_FAST_DEBUG)
    set(CMAKE_CUDA_FLAGS_DEBUG "-g" CACHE STRING "" FORCE)
    message(STATUS "[CT] CUDA Debug: FAST mode (no -G, kernels optimised)")
else()
    message(STATUS "[CT] CUDA Debug: standard mode (-G -g, kernels NON-optimised)")
endif()

# ============================================================
# windeployqt: разворачиваем Qt-плагины и QML для ОБОИХ exe
# ============================================================
if(WIN32)
    find_program(WINDEPLOYQT windeployqt
        HINTS "${Qt6Core_DIR}/../../../bin"
        REQUIRED
    )

    foreach(_tgt IN ITEMS QtCTReconstruction QtCTReconstructionConsumer)
        # Per-config выбор варианта Qt: для Release/RWDI/MinSizeRel — release-DLL
        # (Qt6Core.dll), для Debug — debug-DLL (Qt6Cored.dll). Это критично для
        # переносимости Release-сборки: debug Qt тянет vcruntime140d.dll,
        # которого нет на машинах без Visual Studio.
        add_custom_command(TARGET ${_tgt} POST_BUILD
            COMMAND ${WINDEPLOYQT}
                $<IF:$<CONFIG:Debug>,--debug,--release>
                --qmldir "${CMAKE_CURRENT_SOURCE_DIR}/QtCTReconstruction"
                --no-translations
                --no-system-d3d-compiler
                "$<TARGET_FILE_DIR:${_tgt}>/$<TARGET_FILE_NAME:${_tgt}>"
            COMMENT "📦 Deploying Qt dependencies for ${_tgt} ($<CONFIG>)..."
            VERBATIM
        )

        # Копируем icon.png рядом с .exe (для setWindowIcon в рантайме).
        # Используем CMake-скрипт, чтобы проверка "файл существует" делалась
        # на этапе сборки (а не configure'а) — если пользователь сгенерил PNG
        # ПОСЛЕ cmake configure, иконка всё равно скопируется без re-configure.
        add_custom_command(TARGET ${_tgt} POST_BUILD
            COMMAND ${CMAKE_COMMAND}
                "-DSRC=${CT_APP_ICON_PNG}"
                "-DDST=$<TARGET_FILE_DIR:${_tgt}>/icon.png"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/copy_if_exists.cmake"
            COMMENT "🖼  Copying icon.png next to ${_tgt}.exe (if generated)"
            VERBATIM
        )
    endforeach()

    # Оба exe лежат в одной директории out/build/release/. Их POST_BUILD
    # windeployqt параллельно пишут одни и те же Qt-плагины/QML-файлы → race
    # condition ("Cannot copy ... Destination file exists"). Принудительно
    # сериализуем: Consumer ждёт, пока Producer не закончит линковку + деплой.
    # На повторной сборке windeployqt видит "up to date" → перезаписей нет,
    # без зависимости тоже работает; но первая чистая сборка падает.
    add_dependencies(QtCTReconstructionConsumer QtCTReconstruction)
endif()
