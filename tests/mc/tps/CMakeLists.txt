cmake_minimum_required(VERSION 3.12)
project(tps)

set(GPU ON)

# The extension of the main code should be cpp to compile it using g++
# for CPU version and using nvcc for GPU version.
# It will not be compiled using g++ if the extension is cu
add_executable(tps_env tps_env.cpp)

if (GPU)
    find_package(CUDAToolkit REQUIRED)
    enable_language(CUDA)
    if(NOT DEFINED CMAKE_CUDA_STANDARD)
        set(CMAKE_CUDA_STANDARD 11)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    endif()
    message("Compile using ${CMAKE_CUDA_COMPILER} ${CMAKE_CUDA_COMPILER_VERSION}")

    set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "CUDA architectures")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -w --use_fast_math")
    if(20 IN_LIST CMAKE_CUDA_ARCHITECTURES)
        if (${CMAKE_CUDA_COMPILER_VERSION} VERSION_LESS_EQUAL 9.0.0)
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets") # silence warnings
        endif()
    endif()

    set_property(TARGET tps_env PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
    set_source_files_properties(tps_env.cpp PROPERTIES LANGUAGE CUDA)
else ()
    enable_language(CXX)
    if(NOT DEFINED CMAKE_CXX_STANDARD)
        set(CMAKE_CXX_STANDARD 11)
        set(CMAKE_CXX_STANDARD_REQUIRED ON)
    endif()
    message("Compile using ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_VERSION}")
endif ()

find_package(GDCM REQUIRED)
find_package(ZLIB REQUIRED)

if (APPLE)
    find_library(COREFOUNDATION_LIBRARY CoreFoundation)
endif ()

target_include_directories(tps_env PUBLIC
        ${PROJECT_SOURCE_DIR}
        ${GDCM_INCLUDE_DIRS}
        ../../../
)

target_link_directories(tps_env PUBLIC ${GDCM_DIR})

target_link_libraries(tps_env
        ${COREFOUNDATION_LIBRARY}
        gdcmCommon
        gdcmDSED
        gdcmMEXD
        gdcmjpeg12
        gdcmjpeg8
        gdcmDICT
        gdcmIOD
        gdcmMSFF
        gdcmjpeg16
        ZLIB::ZLIB
        )
