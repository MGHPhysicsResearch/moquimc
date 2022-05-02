#ifndef MQI_COMMON_HPP
#define MQI_COMMON_HPP

/// \file
///
/// A header including CUDA related headers and functions

#if defined(__CUDACC__)

#include <cublas.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
//#include "helper/helper_cuda.h"
//#include "helper/helper_math.h"
#include <cuda_fp16.h>
#include <curand.h>
#include <nvfunctional>
#include <stdio.h>

#define CUDA_HOST_DEVICE __host__ __device__
#define CUDA_HOST __host__
#define CUDA_DEVICE __device__
#define CUDA_GLOBAL __global__
#define CUDA_LOCAL __local__
#define CUDA_SHARED __shared__
#define CUDA_CONSTANT __constant__

#else

#define CUDA_HOST_DEVICE
#define CUDA_HOST
#define CUDA_DEVICE
#define CUDA_GLOBAL
#define CUDA_LOCAL
#define CUDA_SHARED
#define CUDA_CONSTANT
#endif

#include <cmath>
#include <cstdint>
#include <limits>

namespace mqi
{
typedef float phsp_t;

///<
typedef uint64_t cnb_t;
typedef int32_t  ijk_t;

#if defined(__CUDACC__)
//typedef __half density_t;
typedef float density_t;
#else
typedef float density_t;
#endif
const uint16_t   block_limit  = 65535;   //limited to 2^16 //maybe larger?
const uint16_t   thread_limit = 512;
typedef uint32_t key_t;
const key_t      empty_pair = 0xffffffff;

///< enumerate to indicate cell-side
///< XP, XM : a YZ plane at (XP)lus and (XM)inus : 1st axis
///< YP, YM : a YZ plane at (YP)lus and (YM)inus : 2nd axis
///< ZP, ZM : a YZ plane at (ZP)lus and (ZM)inus : 3rd axis
typedef enum
{
    XM             = 0,
    XP             = 1,
    YM             = 2,
    YP             = 3,
    ZM             = 4,
    ZP             = 5,
    NONE_XYZ_PLANE = 6
} cell_side;

typedef enum
{
    MASK   = 0,
    VOLUME = 1
} aperture_type_t;

typedef enum
{
    PER_BEAM    = 0,
    PER_SPOT    = 1,
    PER_PATIENT = 2
} sim_type_t;

typedef enum
{
    APERTURE_CLOSE = 1,
    APERTURE_OPEN  = 2,
    NORMAL_PHYSICS = 3
} transport_type;
const float max_step_global = 1.0;

}   // namespace mqi

#endif
