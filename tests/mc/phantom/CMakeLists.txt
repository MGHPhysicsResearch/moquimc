cmake_minimum_required(VERSION 3.10)
project(env)

set(GPU ON)

# The extension of the main code should be cpp to compile it using g++
# for CPU version and using nvcc for GPU version.
# It will not be compiled using g++ if the extension is cu
add_executable(phantom_env phantom_env.cpp)
add_definitions(-D__PHYSICS_DEBUG__)


if (GPU)
    message("Compile using nvcc")
    find_package(CUDAToolkit REQUIRED)
    enable_language(CUDA)
    set_property(TARGET phantom_env PROPERTY CUDA_ARCHITECTURES 75)
    set(CMAKE_CUDA_COMPILER "nvcc" CACHE STRING "Path to nvcc CUDA compiler")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -w --use_fast_math")
    set_source_files_properties(phantom_env.cpp PROPERTIES LANGUAGE CUDA)

else ()
    message("Compile using g++")
    enable_language(CXX)
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS ON)
    set(CMAKE_CXX_COMPILER g++)
endif ()

find_package(GDCM REQUIRED)
include(${GDCM_USE_FILE})

if (APPLE)
    find_library(COREFOUNDATION_LIBRARY CoreFoundation)
endif ()

include_directories(
        ${PROJECT_SOURCE_DIR}
        ${GDCM_INCLUDE_DIRS}
        ../../../
        /usr/local/cuda/include
)

link_directories(${GDCM_DIR})

target_link_libraries(phantom_env
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
        )
