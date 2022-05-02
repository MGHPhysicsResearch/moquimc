#ifndef MQI_UTILS_H
#define MQI_UTILS_H

/// \file
/// Header file containing general functions useful for several classes
/// \see http://www.martinbroadhurst.com/how-to-trim-a-stdstring.html
/// \see https://stackoverflow.com/questions/1798112/removing-leading-and-trailing-spaces-from-a-string

#include <array>
#include <chrono>
#include <map>
#include <moqui/base/mqi_common.hpp>
#include <string>
#include <tuple>
#include <vector>

namespace mqi
{

///< Remove white-space on right
inline std::string
trim_right_copy(const std::string& s, const std::string& delimiters = " \f\n\r\t\v\0\\") {
    return s.substr(0, s.find_last_not_of(delimiters) + 1);
}

///< Remove white-space on left
inline std::string
trim_left_copy(const std::string& s, const std::string& delimiters = " \f\n\r\t\v\0\\") {
    return s.substr(s.find_first_not_of(delimiters));
}

///< Remove white-space on left and right
inline std::string
trim_copy(const std::string& s, const std::string& delimiters = " \f\n\r\t\v\0\\") {
    return trim_left_copy(trim_right_copy(s, delimiters), delimiters);
}

/// Interpolates values in map
/// \tparam T is type of values
/// \tparam S is size of columns
/// \param db map of dataset
/// \param x position value where interpolated value is calculated at
/// \param y_col column index of map for y-values to be used for interpolation
template<typename T, size_t S>
inline T
interp_linear(const std::map<T, std::array<T, S>>& db, const T x, const size_t y_col = 0) {
    auto it_up = db.upper_bound(x);
    if (it_up == db.end()) { return ((--it_up)->second)[y_col]; }
    if (it_up == db.begin()) { return (it_up->second)[y_col]; }
    T    x1      = it_up->first;
    T    y1      = (it_up->second)[y_col];
    auto it_down = --it_up;
    T    x0      = it_down->first;
    T    y0      = (it_down->second)[y_col];
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0);
}

/// Interpolates values in vector
/// \tparam T is type of values
/// \tparam S is size of columns
/// \param db is vector for dataset
/// \param x position value where interpolated value is calculated at
/// \param x_col column index of map for x-values to be used for interpolation
/// \param y_col column index of map for y-values to be used for interpolation
template<typename T, size_t S>
inline T
interp_linear(const std::vector<std::array<T, S>>& db,
              const T                              x,
              const size_t                         x_col = 0,
              const size_t                         y_col = 1) {
    if (x <= db[0][x_col]) return db[0][y_col];
    for (size_t i = 1; i < db.size() - 1; ++i) {
        if (x <= db[i][x_col]) {

            T x0 = db[i - 1][x_col];
            T x1 = db[i][x_col];
            T y0 = db[i - 1][y_col];
            T y1 = db[i][y_col];

            return y0 + (x - x0) * (y1 - y0) / (x1 - x0);
        }
    }

    return db[db.size() - 1][y_col];
}

/// Interpolate table lambda function
/// \param vector_X The array of x coordinates
/// \param vector_Y The array of y coordinates
/// \param x the ordinate to evaluate
/// \param npoints the number of coordinates in the table
/// \param order the interpolation polynom order
/// \return the y value corresponding to the x ordinate
/// \note from gpmc code
inline float
TableInterpolation(float* const vector_X,
                   float* const vector_Y,
                   const float  x,
                   const int    npoints,
                   int          order = 4) {
    float result;
    // check order of interpolation
    if (order > npoints) order = npoints;
    // if x is ouside the vector_X[] interval
    if (x <= vector_X[0]) return result = vector_Y[0];
    if (x >= vector_X[npoints - 1]) return result = vector_Y[npoints - 1];
    // loop to find j so that x[j-1] < x < x[j]
    int j = 0;
    while (j < npoints) {
        if (vector_X[j] >= x) break;
        j++;
    }
    // shift j to correspond to (npoint-1)th interpolation
    j = j - order / 2;
    // if j is ouside of the range [0, ... npoints-1]
    if (j < 0) j = 0;
    if (j + order > npoints) j = npoints - order;
    result = 0.0;
    // Allocate enough space for any table we'd like to read.
    float* lambda = new float[npoints];
    for (int is = j; is < j + order; is++) {
        lambda[is] = 1.0;
        for (int il = j; il < j + order; il++) {
            if (il != is)
                lambda[is] = lambda[is] * (x - vector_X[il]) / (vector_X[is] - vector_X[il]);
        }
        result += vector_Y[is] * lambda[is];
    }
    delete[] lambda;
    return result;
}

///< This will return start & range of jobs for given thread-id of N-threads.
///< For example, if we have 11 histories and two threads to process,
///< this function will return 0,6 for thread-0 and 6,5 for thread-1.
CUDA_DEVICE
vec2<uint32_t>
start_and_length(const uint32_t& n_threads, const uint32_t& n_jobs, const uint32_t& thread_id) {

    uint32_t quotient  = n_jobs / n_threads;
    uint32_t remainder = n_jobs % n_threads;

    mqi::vec2<uint32_t> ret;
    ret.x = quotient * thread_id + ((thread_id >= remainder) ? remainder : thread_id);
    ret.y = quotient + 1 * (thread_id < remainder);
    return ret;
}

///< this is equivalent to std::lower_bound
///< "arr" is assumed to be sorted
CUDA_HOST_DEVICE
int32_t
lower_bound_cpp(const int32_t* arr, const int32_t& len, const int32_t& value) {
    int32_t first = 0;
    int32_t count = len;
    int32_t step;

    int32_t it;
    while (count > 0) {

        it   = first;
        step = count / 2;
        it += step;

        if (arr[it] <= value) {
            first = ++it;
            count -= step + 1;
        } else
            count = step;
    }
    return first;
}

}   // namespace mqi

#endif
