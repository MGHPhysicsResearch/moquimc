#ifndef MQI_VEC_H
#define MQI_VEC_H

/// \file
///
/// RT-Ion vector for 2D, 3D, and 4D.

#include <array>
#include <cmath>
#include <iostream>
#include <moqui/base/mqi_common.hpp>

namespace mqi
{

/// \class vec2
/// a vector with two position
/// \tparam T type of vector elements
template<typename T>
class vec2
{
public:
    //
    T x;
    T y;

    CUDA_HOST_DEVICE
    vec2(vec2& ref) {
        x = ref.x;
        y = ref.y;
    }

    CUDA_HOST_DEVICE
    vec2() : x(0), y(0) {
        ;
    }

    CUDA_HOST_DEVICE
    vec2(T a, T b) : x(a), y(b) {
        ;
    }

    CUDA_HOST_DEVICE
    vec2(const vec2& ref) : x(ref.x), y(ref.y) {
        ;
    }

    CUDA_HOST
    vec2(const std::array<T, 2>& ref) : x(ref[0]), y(ref[1]) {
        ;
    }
    //#endif

    CUDA_HOST_DEVICE
    ~vec2() {
        ;
    }

    CUDA_HOST_DEVICE
#if defined(__CUDACC__)
    //sqrtf : for float, sqrtg: for double
    T
    norm() const {
        return sqrtf(x * x + y * y);
    }
#else
    T
    norm() const {
        return std::sqrt(x * x + y * y);
    }
#endif

    CUDA_HOST_DEVICE
    T
    dot(const vec2<T>& v) const {
        return x * v.x + y * v.y;
    }

    //cross product of 2d vec is scalar
    CUDA_HOST_DEVICE
    T
    cross(const vec2<T>& v) const {
        return x * v.y - y * v.x;
    }

    CUDA_HOST_DEVICE
    vec2<T>&
    operator=(const vec2<T>& r) {
        x = r.x;
        y = r.y;
        return *this;
    }

    CUDA_HOST_DEVICE
    vec2<T>
    operator+(const vec2<T>& r) const {
        return vec2<T>(x + r.x, y + r.y);
    }

    CUDA_HOST_DEVICE
    vec2<T>
    operator-(const vec2<T>& r) const {
        return vec2<T>(x - r.x, y - r.y);
    }

    CUDA_HOST_DEVICE
    vec2<T>
    operator*(const T& r) const {
        return vec2<T>(x * r, y * r);
    }

    template<typename R>
    vec2<T>
    operator*(const R& r) const {
        return vec2<T>(x * r, y * r);
    }

    CUDA_HOST_DEVICE
    vec2<T>
    operator*(const vec2<T>& r) const {
        return vec2<T>(x * r.x, y * r.y);
    }

    CUDA_HOST_DEVICE
    void
    dump() const {
        printf("(x,y): (%f, %f)\n", x, y);
    }
};

/// \class vec3
/// a vector with three position
/// \tparam T type of vector elements
//this is a copy of HepRotation.
template<typename T>
class vec3
{
public:
    //
    T x;
    T y;
    T z;

    CUDA_HOST_DEVICE
    vec3(vec3& ref) {
        x = ref.x;
        y = ref.y;
        z = ref.z;
    }

    CUDA_HOST_DEVICE
    vec3() : x(0), y(0), z(0) {
        ;
    }

    CUDA_HOST_DEVICE
    vec3(T a, T b, T c) : x(a), y(b), z(c) {
        ;
    }

    CUDA_HOST_DEVICE
    vec3(const vec3& ref) : x(ref.x), y(ref.y), z(ref.z) {
        ;
    }

    CUDA_HOST
    vec3(const std::array<T, 3>& ref) : x(ref[0]), y(ref[1]), z(ref[2]) {
        ;
    }

    CUDA_HOST_DEVICE
    vec3(const T* ref) : x(ref[0]), y(ref[1]), z(ref[2]) {
        ;
    }

    CUDA_HOST_DEVICE
    ~vec3() {
        ;
    }

    CUDA_HOST_DEVICE
#if defined(__CUDACC__)
    //sqrtf : for float, sqrtg: for double
    T
    norm() const {
        return sqrtf(x * x + y * y + z * z);
    }
#else
    T
    norm() const {
        return std::sqrt(x * x + y * y + z * z);
    }
#endif

    CUDA_HOST_DEVICE
#if defined(__CUDACC__)
    //sqrtf : for float, sqrtg: for double
    void
    normalize() {
        T n = sqrtf(x * x + y * y + z * z);
        x /= n;
        y /= n;
        z /= n;
    }
#else
    void
    normalize() {
        T n = std::sqrt(x * x + y * y + z * z);
        x /= n;
        y /= n;
        z /= n;
    }
#endif

    CUDA_HOST_DEVICE
    vec3<T>
    operator+(const vec3<T>& r) const {
        return vec3<T>(x + r.x, y + r.y, z + r.z);
    }

    CUDA_HOST_DEVICE
    vec3<T>
    operator-(const vec3<T>& r) const {
        return vec3<T>(x - r.x, y - r.y, z - r.z);
    }

    CUDA_HOST_DEVICE
    vec3<T>
    operator*(const T& r) const {
        return vec3<T>(x * r, y * r, z * r);
    }

    CUDA_HOST_DEVICE
    vec3<T>
    operator*(const T& r) {
        return vec3<T>(x * r, y * r, z * r);
    }

    CUDA_HOST_DEVICE
    vec3<T>
    operator/(const T& r) const {
        return vec3<T>(x / r, y / r, z / r);
    }

    CUDA_HOST_DEVICE
    T
    dot(const vec3<T>& v) const {
        return x * v.x + y * v.y + z * v.z;
    }

    CUDA_HOST_DEVICE
    vec3<T>
    cross(const vec3<T>& r) const {
        return vec3<T>(y * r.z - z * r.y, z * r.x - x * r.z, x * r.y - y * r.x);
    }

    CUDA_HOST_DEVICE
    vec3<T>&
    operator=(const vec3<T>& r) {
        x = r.x;
        y = r.y;
        z = r.z;
        return *this;
    }

    CUDA_HOST_DEVICE
    vec3<T>&
    operator+=(const vec3<T>& r) {
        x += r.x;
        y += r.y;
        z += r.z;
        return *this;
    }

    CUDA_HOST_DEVICE
    void
    dump() const {
#if defined(__CUDACC__)
        printf("x,y,z: (%f, %f, %f)\n", x, y, z);
#else
        std::cout << "x,y,z: (" << x << ", " << y << ", " << z << ") " << std::endl;
#endif
    }
};

/// \class vec4
/// a vector with four position
/// \tparam T type of vector elements
//this is a copy of HepRotation.
template<typename T>
class vec4
{
public:
    //
    T x;
    T y;
    T z;
    T s;   //scale

    CUDA_HOST_DEVICE
    vec4(vec4& ref) {
        x = ref.x;
        y = ref.y;
        z = ref.z;
        s = ref.s;
    }

    CUDA_HOST_DEVICE
    vec4() : x(0), y(0), z(0), s(0) {
        ;
    }

    CUDA_HOST_DEVICE
    vec4(T a, T b, T c, T d) : x(a), y(b), z(c), s(d) {
        ;
    }

    CUDA_HOST_DEVICE
    vec4(const vec4& ref) : x(ref.x), y(ref.y), z(ref.z), s(ref.s) {
        ;
    }

    CUDA_HOST_DEVICE
    vec4(const T* ref) : x(ref[0]), y(ref[1]), z(ref[2]), s(ref[3]) {
        ;
    }

    CUDA_HOST
    vec4(const std::array<T, 4>& ref) : x(ref[0]), y(ref[1]), z(ref[2]), s(ref[3]) {
        ;
    }
    //#endif

    CUDA_HOST_DEVICE
    ~vec4() {
        ;
    }

    CUDA_HOST_DEVICE
#if defined(__CUDACC__)
    //sqrtf : for float, sqrtg: for double
    T
    norm() const {
        return sqrtf(x * x + y * y + z * z + s * s);
    }
#else
    T
    norm() const {
        return std::sqrt(x * x + y * y + z * z + s * s);
    }
#endif

    CUDA_HOST_DEVICE
    vec4<T>
    operator+(const vec4<T>& r) const {
        return vec4<T>(x + r.x, y + r.y, z + r.z, s + r.s);
    }

    CUDA_HOST_DEVICE
    vec4<T>
    operator-(const vec4<T>& r) const {
        return vec4<T>(x - r.x, y - r.y, z - r.z, s - r.s);
    }

    CUDA_HOST_DEVICE
    vec4<T>
    operator*(const T& r) const {
        return vec4<T>(x * r, y * r, z * r, s * r);
    }

    CUDA_HOST_DEVICE
    void
    dump() const {
        printf("(x,y,z,s): (%f, %f, %f)\n", x, y, z, s);
    }
};

}   // namespace mqi

#endif
