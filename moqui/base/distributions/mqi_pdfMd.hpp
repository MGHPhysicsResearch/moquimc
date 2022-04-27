
#ifndef MQI_PDFMD_H
#define MQI_PDFMD_H

/// \file
/// 
/// Distribution functions (meta-header file for all distributions)
 
#include <random>
#include <functional>
#include <queue>
#include <iostream>
#include <fstream>
#include <random>
#include <chrono>
#include <functional>
#include <array>

#include <moqui/base/mqi_math.hpp>
#include <moqui/base/mqi_vec.hpp>
#include <moqui/base/mqi_matrix.hpp>


namespace mqi{

/// \class pdf_Md
///
/// M-dimensional probability distribution function (pdf).
/// \tparam T type of return value
/// \tparam M size of return values
/// pure virtual class
template<typename T, std::size_t M>
class pdf_Md{
protected:
    std::array<T,M> mean_  ; ///< M means 
    std::array<T,M> sigma_ ; ///< M sigmas
public:

    /// Constructor to fill mean_ and sigma_ 
    CUDA_HOST_DEVICE
    pdf_Md(
        std::array<T,M>& m, 
        std::array<T,M>& s)
    {
        for(std::size_t i=0; i < M ; ++i){
            mean_[i] = m[i];
            sigma_[i] = s[i];
        }
    }

    /// Constructor to fill const mean_ and const sigma_ 
    CUDA_HOST_DEVICE
    pdf_Md(
        const std::array<T,M>& m, 
        const std::array<T,M> &s)
    {
        for(std::size_t i=0; i < M ; ++i){
            mean_[i] = m[i];
            sigma_[i] = s[i];
        }
    }
    
    /// Prints out means and sigmas
    CUDA_HOST_DEVICE
    void 
    dump(){
        for(size_t i=0 ; i < M ; ++i)
        std::cout<< "mean: " << mean_[i] << ", sigma: " << sigma_[i] << std::endl;
    }
    
    /// Destructor
    CUDA_HOST_DEVICE
    ~pdf_Md(){;}


    /// '()' operator overloading to act like a function. 
    CUDA_HOST_DEVICE
    virtual 
    std::array<T,M>
    operator()(std::default_random_engine* rng) = 0;

};


}

#endif

