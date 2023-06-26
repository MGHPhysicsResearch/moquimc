cmake_minimum_required(VERSION 3.12)
project(moqui)

find_package(GDCM REQUIRED)
find_package(ZLIB REQUIRED)

if (APPLE)
    find_library(COREFOUNDATION_LIBRARY CoreFoundation)
endif ()

### Set release mode as default build type
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")# "MinSizeRel" "RelWithDebInfo"
endif()

if (CMAKE_BUILD_TYPE STREQUAL Debug)
add_definitions(-D__PHYSICS_DEBUG__)
endif()

set(USE_CUDA ON)

# The extension of the main code should be cpp to compile it using g++
# for CPU version and using nvcc for GPU version.
# It will not be compiled using g++ if the extension is cu
if (USE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    enable_language(CUDA)
    if(NOT DEFINED CMAKE_CUDA_STANDARD)
        set(CMAKE_CUDA_STANDARD 11)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    endif()
    message(STATUS "Compile using ${CMAKE_CUDA_COMPILER} ${CMAKE_CUDA_COMPILER_VERSION}")

    set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "CUDA architectures")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -w --use_fast_math")
    if(20 IN_LIST CMAKE_CUDA_ARCHITECTURES)
        if (${CMAKE_CUDA_COMPILER_VERSION} VERSION_LESS_EQUAL 9.0.0)
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets") # silence warnings
            string(REGEX REPLACE "([\\/\\-]O)3" "\\10" CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE}")
        endif()
    endif()

    ### When in debug mode: separate compilation and device debugging
    # Remember setting in QtCreator cuda-gdb as debugger in the Compiler kit
    if (CMAKE_BUILD_TYPE STREQUAL Debug)
        #set(SEPARATE_COMPILATION ON CACHE BOOL "Compile each source file separately. Results in less optimized code but good for debugging.")
        #set(DEVICE_DEBUGGING ON CACHE BOOL "Debug device code.")
        set(CMAKE_CUDA_FLAGS_DEBUG "-G -g -Xcompiler -Wall ${CMAKE_CUDA_FLAGS_DEBUG}")#-rdynamic
        #TODO, if you want profiling at high speed, use -lineinfo instead of -G
        #See also --source-in-ptx https://gitlab.kitware.com/cmake/cmake/-/issues/19017
    endif()

    #~ if (SEPARATE_COMPILATION)
    #set(CUDA_SEPARABLE_COMPILATION ON)
    #set_property(TARGET myexe PROPERTY CUDA_SEPARABLE_COMPILATION ON)

else ()
    enable_language(CXX)
    if(NOT DEFINED CMAKE_CXX_STANDARD)
        set(CMAKE_CXX_STANDARD 11)
        set(CMAKE_CXX_STANDARD_REQUIRED ON)
    endif()
    message(STATUS "Compile using ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_VERSION}")
endif ()

add_subdirectory(moqui)

option(ENABLE_TESTS "Enables end-to-end integration tests using CTest framework." ON)
if(ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()


