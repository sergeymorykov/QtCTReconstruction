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

# Source files
set(SOURCES
    QtCTReconstruction/main.cpp
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
    QtCTReconstruction/CUDAPureBackend.cu
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

qt_add_executable(${PROJECT_NAME} MANUAL_FINALIZATION ${SOURCES})

# --- Зависимости, связываемые с FFT-бэкендом ---
if(FFT_BACKEND STREQUAL "FFTW" AND FFTW3F_FOUND)
    target_sources(${PROJECT_NAME} PRIVATE
        QtCTReconstruction/FFTBackendFFTW.cpp
    )
    target_compile_definitions(${PROJECT_NAME} PRIVATE CT_FFT_BACKEND_FFTW)
    target_include_directories(${PROJECT_NAME} PRIVATE ${FFTW3F_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${FFTW3F_LIBRARIES})
endif()

# --- Serial backend ---
if(BUILD_SERIAL_BACKEND)
    target_sources(${PROJECT_NAME} PRIVATE QtCTReconstruction/SerialBackend.cpp)
    target_compile_definitions(${PROJECT_NAME} PRIVATE CT_HAS_SERIAL_BACKEND)
    message(STATUS "[CT] SerialBackend: ENABLED")
else()
    message(STATUS "[CT] SerialBackend: DISABLED (Release build)")
endif()

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE QtCTReconstruction)

# Dependencies
target_link_libraries(${PROJECT_NAME} PRIVATE
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

# === FIX FOR MSVC: LLVM OpenMP + C++ Exception Semantics ===
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/openmp:llvm>
        $<$<COMPILE_LANGUAGE:CXX>:/openmp:experimental>
        $<$<COMPILE_LANGUAGE:CXX>:/EHsc>
    )
endif()

# QML files (Qt 6 resource system)
qt_add_qml_module(${PROJECT_NAME}
    URI QtCTReconstruction
    VERSION 1.0
    QML_FILES
        QtCTReconstruction/main.qml
        QtCTReconstruction/PointCloudWindow.qml
        QtCTReconstruction/ImageWithAxes.qml
)

# Configurations for Debug/Release specifics if needed in C++
target_compile_definitions(${PROJECT_NAME} PRIVATE 
    $<$<CONFIG:Debug>:CT_DEBUG_MODE>
    $<$<CONFIG:Release>:CT_RELEASE_MODE>
)

qt_import_qml_plugins(${PROJECT_NAME})
qt_finalize_executable(${PROJECT_NAME})

if(MSVC)
    add_compile_options(/utf-8 /Zc:preprocessor)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler /Zc:preprocessor")
endif()

if(WIN32)
    find_program(WINDEPLOYQT windeployqt 
        HINTS "${Qt6Core_DIR}/../../../bin" 
        REQUIRED
    )
    
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${WINDEPLOYQT} 
            --debug 
            --qmldir "${CMAKE_CURRENT_SOURCE_DIR}/QtCTReconstruction"
            --no-translations 
            --no-system-d3d-compiler
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>/$<TARGET_FILE_NAME:${PROJECT_NAME}>"
        COMMENT "📦 Deploying Qt dependencies with windeployqt..."
        VERBATIM
    )
endif()
