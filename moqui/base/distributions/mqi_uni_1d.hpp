
#ifndef MQI_UNI_1D_H
#define MQI_UNI_1D_H

/// \file
/// 
/// Distribution functions (meta-header file for all distributions)

#include <moqui/base/distributions/mqi_pdfMd.hpp>

namespace mqi{
/// \class uni_1d
///
/// 1-dimensional uniform pdf.
/// \tparam T type of return value
template<typename T>
class uni_1d : public pdf_Md<T,1> {
public:

//    std::default_random_engine  gen_        ; ///< C++ random engine
    std::uniform_real_distribution<T> func_ ; ///< C++ distribution function

    /// Constructor to copy mean and sigma and initialize uniform_distributions
    /// In this pdf, mean[0] and sigma[0] will be used for ranges
    /// e.g., a random variable will be sampled from mean[0] and sigma[0]
    CUDA_HOST_DEVICE
    uni_1d(
        std::array<T,1>& m, 
        std::array<T,1>& s) 
    : pdf_Md<T,1>(m,s)
    {
        #if !defined(__CUDACC__)
        func_ = std::uniform_real_distribution<T>(m[0], s[0]); 
        #endif
    }
    
    /// Constructor to copy mean and sigma and initialize uniform_distributions
    CUDA_HOST_DEVICE
    uni_1d(
        const std::array<T,1>& m, 
        const std::array<T,1>& s)
    : pdf_Md<T,1>(m,s)
    {
        #if !defined(__CUDACC__)
        func_ = std::uniform_real_distribution<T>(m[0], s[0]); 
        #endif
    }

    /// Returns value sampled from 1-d uniform distribution
    CUDA_HOST_DEVICE
    virtual 
    std::array<T,1>
    operator()(std::default_random_engine* rng){
        return {func_(*rng)};
    };

};

}
#endif