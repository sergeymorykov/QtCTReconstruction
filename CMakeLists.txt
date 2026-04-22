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
    QtCTReconstruction/PointCloudGeometry.cpp
    QtCTReconstruction/OptimizedBackprojectionCPU.cpp
    QtCTReconstruction/ProjectionGeometry.cpp
)

qt_add_executable(${PROJECT_NAME} MANUAL_FINALIZATION ${SOURCES})

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
