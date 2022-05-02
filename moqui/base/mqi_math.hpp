#ifndef MQI_MATH_HPP
#define MQI_MATH_HPP

/// \file
///
/// A header including CUDA related headers and functions

#include <moqui/base/mqi_common.hpp>

#include <cmath>
#include <mutex>
#include <random>

namespace mqi
{

//const float near_zero = 0.0000001;
const float near_zero          = 1e-7;
const float min_step           = 1e-3;
const float geometry_tolerance = 1e-3;

///< TODO: CUDA m_inf
const float m_inf = -1.0 * HUGE_VALF;
const float p_inf = HUGE_VALF;

template<typename T>
CUDA_DEVICE inline T
intpl1d(T x, T x0, T x1, T y0, T y1) {
    return (x1 == x0) ? y0 : y0 + (x - x0) * (y1 - y0) / (x1 - x0);
}

template<typename T>
CUDA_DEVICE T
mqi_ln(T s);

template<typename T>
CUDA_HOST_DEVICE T
mqi_sqrt(T s);

template<typename T>
CUDA_DEVICE T
mqi_pow(T s, T p);

template<typename T>
CUDA_DEVICE T
mqi_exp(T s);

template<typename T>
CUDA_DEVICE T
mqi_acos(T s);

template<typename T>
CUDA_DEVICE T
mqi_cos(T s);

template<typename T>
CUDA_DEVICE T
mqi_sin(T s);

template<typename T>
CUDA_DEVICE T
mqi_abs(T s);

template<typename T>
CUDA_HOST_DEVICE T
mqi_round(T s);

template<typename T>
CUDA_HOST_DEVICE T
mqi_floor(T s);

template<typename T>
CUDA_HOST_DEVICE T
mqi_ceil(T s);

template<typename T>
CUDA_HOST_DEVICE bool
mqi_isnan(T s);

/*
template<typename T>
CUDA_DEVICE
T mqi_inf();
*/

///< To make template for both return type and argument.
///< 1. return type template
template<class T>
struct rnd_return {
    typedef T type;
};

template<>
struct rnd_return<float> {
    typedef float type;
};

template<>
struct rnd_return<double> {
    typedef double type;
};

///< 2. distribution funtion template.

///< normal
template<class T, class S>
CUDA_DEVICE typename rnd_return<T>::type
mqi_normal(S* rng, T avg, T sig) {
    return T();
}

///< uniform
template<class T, class S>
CUDA_DEVICE typename rnd_return<T>::type
mqi_uniform(S* rng) {
    return T();
}

///< exponetial distribution
template<class T, class S>
CUDA_DEVICE typename rnd_return<T>::type
mqi_exponential(S* rng, T avg, T up) {
    return T();
}

#if defined(__CUDACC__)

///< specialization of template functions
//Nautual log
template<>
float
mqi_ln(float s) {
    return logf(s);
}
template<>
double
mqi_ln(double s) {
    return log(s);
}

///< sqrt
template<>
float
mqi_sqrt(float s) {
    return sqrtf(s);
}
template<>
double
mqi_sqrt(double s) {
    return sqrt(s);
}

///< power
template<>
float
mqi_pow(float s, float p) {
    return powf(s, p);
}
template<>
double
mqi_pow(double s, double p) {
    return pow(s, p);
}

///< exponential
template<>
float
mqi_exp(float s) {
    return expf(s);
}
template<>
double
mqi_exp(double s) {
    return exp(s);
}

///< acos
template<>
float
mqi_acos(float s) {
    return acosf(s);
}
template<>
double
mqi_acos(double s) {
    return acos(s);
}

///< cos
template<>
float
mqi_cos(float s) {
    return cosf(s);
}
template<>
double
mqi_cos(double s) {
    return cos(s);
}
///< sin
template<>
float
mqi_sin(float s) {
    return sinf(s);
}
template<>
double
mqi_sin(double s) {
    return sin(s);
}

template<>
float
mqi_abs(float s) {
    return abs(s);
}
template<>
double
mqi_abs(double s) {
    return abs(s);
}

template<>
float
mqi_round(float s) {
    return roundf(s);
}
template<>
double
mqi_round(double s) {
    return round(s);
}

template<>
float
mqi_floor(float s) {
    return floorf(s);
}
template<>
double
mqi_floor(double s) {
    return floor(s);
}

template<>
float
mqi_ceil(float s) {
    return ceilf(s);
}
template<>
double
mqi_ceil(double s) {
    return ceil(s);
}

template<>
bool
mqi_isnan(float s) {
    bool t = isnan(s);
    return t;
}
template<>
bool
mqi_isnan(double s) {
    bool t = isnan(s);
    return t;
}

//random number status per thread. each status is initialized by master seed.
//the parameters to be passed are currand_status for CUDA and random_engine for C++
//curand_status == mqi_rng ;
typedef curandState_t mqi_rng;

template<>
float
mqi_uniform<float>(mqi_rng* rng) {
    return curand_uniform(rng);
}

template<>   //template<class S = curandState>
double
mqi_uniform<double>(mqi_rng* rng) {
    return curand_uniform_double(rng);
}

template<>
float
mqi_normal<float>(mqi_rng* rng, float avg, float sig) {
    return curand_normal(rng) * sig + avg;
}

template<>
double
mqi_normal<double>(mqi_rng* rng, double avg, double sig) {
    return curand_normal_double(rng) * sig + avg;
}

template<>
float
mqi_exponential<float>(mqi_rng* rng, float avg, float up) {
    float x;
    do {
        x = -1.0 / avg * logf(1.0 - curand_uniform(rng));   //0, up
    } while (x > up || mqi::mqi_isnan(x));

    return x;
}

template<>
double
mqi_exponential<double>(mqi_rng* rng, double avg, double up) {
    double x;
    do {
        x = -1.0 / avg * log(1.0 - curand_uniform(rng));   //0, up
    } while (x > up || mqi::mqi_isnan(x));
    return x;
}

#else

//Natural log. C++ casts float to double. they have same implementation.
template<>
float
mqi_ln(float s) {
    return std::log(s);
}

template<>
double
mqi_ln(double s) {
    return std::log(s);
}

template<>
float
mqi_sqrt(float s) {
    return std::sqrt(s);
}

template<>
double
mqi_sqrt(double s) {
    return std::sqrt(s);
}

template<>
float
mqi_pow(float s, float p) {
    return std::pow(s, p);
}

template<>
double
mqi_pow(double s, double p) {
    return std::pow(s, p);
}

///< exponential
template<>
float
mqi_exp(float s) {
    return std::exp(s);
}
template<>
double
mqi_exp(double s) {
    return std::exp(s);
}

///< acos
template<>
float
mqi_acos(float s) {
    return std::acos(s);
}
template<>
double
mqi_acos(double s) {
    return std::acos(s);
}

template<>
float
mqi_cos(float s) {
    return std::cos(s);
}
template<>
double
mqi_cos(double s) {
    return std::cos(s);
}

template<>
float
mqi_abs(float s) {
    return std::abs(s);
}
template<>
double
mqi_abs(double s) {
    return std::abs(s);
}

///< round
template<>
float
mqi_round(float s) {
    return std::roundf(s);
}
template<>
double
mqi_round(double s) {
    return std::round(s);
}

////< floor

template<>
float
mqi_floor(float s) {
    return std::floor(s);
}
template<>
double
mqi_floor(double s) {
    return std::floor(s);
}

////< ceil
template<>
float
mqi_ceil(float s) {
    return std::ceil(s);
}
template<>
double
mqi_ceil(double s) {
    return std::ceil(s);
}

////< isnan
template<>
bool
mqi_isnan(float s) {
    return std::isnan(s);
}
template<>
bool
mqi_isnan(double s) {
    return std::isnan(s);
}

//compile: ERROR
typedef std::default_random_engine mqi_rng;
//typedef std::mt19937 mqi_rng;

template<>
float
mqi_uniform<float>(mqi_rng* rng) {
    std::uniform_real_distribution<float> dist;
    return dist(*rng);
}

template<>
double
mqi_uniform<double>(mqi_rng* rng) {
    std::uniform_real_distribution<double> dist;
    return dist(*rng);
}

template<>
float
mqi_normal<float>(mqi_rng* rng, float avg, float sig) {
    std::normal_distribution<float> dist(avg, sig);
    return dist(*rng);
}

template<>
double
mqi_normal<double>(mqi_rng* rng, double avg, double sig) {
    std::normal_distribution<double> dist(avg, sig);
    return dist(*rng);
}

template<>
float
mqi_exponential<float>(mqi_rng* rng, float avg, float up) {
    float                                x;
    std::exponential_distribution<float> dist(avg);
    x = dist(*rng);
    //    do {
    //        x = dist(*rng);
    //    } while (x > up || x <= 0);
    return x;
}

template<>
double
mqi_exponential<double>(mqi_rng* rng, double avg, double up) {
    double                                x;
    std::exponential_distribution<double> dist(avg);
    do {
        x = dist(*rng);
    } while (x > up || x <= 0);
    return x;
}

#endif

}   // namespace mqi

#endif
