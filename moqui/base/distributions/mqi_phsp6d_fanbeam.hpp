
#ifndef MQI_PHSP_6D_FANBEAM_H
#define MQI_PHSP_6D_FANBEAM_H

/// \file
/// 
/// Distribution functions (meta-header file for all distributions)
 
#include <moqui/base/distributions/mqi_pdfMd.hpp>

namespace mqi{

/// \class phsp_6d_fanbeam
///
/// 6-dimensional uniform pdf for phase-space variables for a fanbeam (wide)
//  phase-space variables are position (x,y,z) and direction (x',y',z') between x_max, x_min and y_max and y_min, etc.
/// \tparam T type of return value
template<typename T>
class phsp_6d_fanbeam : public pdf_Md<T,6> {

/// correlations for x-x' and y-y' respectively
std::array<T,2> rho_; 
/// Source to Axis Distance for x and y.
std::array<T,2> SAD_; 

public:
    
    /// Random engine and distribution function
//    std::default_random_engine        gen_ ;
    std::normal_distribution<T>       func_;

    /// uniform distributions to place x and y position
    /// positions are determined with SAD
    std::uniform_real_distribution<T> unifx_;  
    std::uniform_real_distribution<T> unify_;

    /// Constructor to initialize distribution parameters
    CUDA_HOST_DEVICE
    phsp_6d_fanbeam(
        std::array<T,6>& m,  ///< [x_min, x_max, y_min, y_max, z_min, z_max] 
        std::array<T,6>& s,  ///< [sig_x, sig_y, sig_z, sig_x', sig_y', sig_z']
        std::array<T,2>& r,  ///< rho_ correlation of x,x' and y,y'
        std::array<T,2>& o   ///< [SADx, SADy]
    ) : pdf_Md<T,6>(m,s), 
	    rho_(r), 
	    SAD_(o) 
    {
    	//#if !defined(__CUDACC__)
//        gen_.seed(std::chrono::system_clock::now().time_since_epoch().count());
        unifx_ = std::uniform_real_distribution<T>(m[0], m[1]); 
        unify_ = std::uniform_real_distribution<T>(m[2], m[3]); 
        func_ = std::normal_distribution<T>(0, 1); 
        //#endif
    }
    
    /// Constructor to initialize distribution parameters
    CUDA_HOST_DEVICE
    phsp_6d_fanbeam(
	    const std::array<T,6>& m,  ///< [x_min, x_max, y_min, y_max, z_min, z_max]     
        const std::array<T,6>& s,  ///< [sig_x, sig_y, sig_z, sig_x', sig_y', sig_z']
        const std::array<T,2>& r,  ///< rho_ correlation of x,x' and y,y'
        const std::array<T,2>& o   ///< [SADx, SADy]
    ) : pdf_Md<T,6>(m,s), 
	    rho_(r), 
	    SAD_(o) 
    {
	//#if !defined(__CUDACC__)
//        gen_.seed(std::chrono::system_clock::now().time_since_epoch().count());
        unifx_ = std::uniform_real_distribution<T>(m[0], m[1]); 
        unify_ = std::uniform_real_distribution<T>(m[2], m[3]); 
        func_ = std::normal_distribution<T>(0, 1); 
        //#endif
    }

    /// Constructor to initialize distribution parameters
    /// phase-space variables are sampled same in phsp_6d but they are spread along x and y.
    CUDA_HOST_DEVICE
    virtual 
    std::array<T,6>
    operator()(std::default_random_engine* rng)
    {
	    auto x = unifx_(*rng) ;
	    auto y = unify_(*rng) ;
	
	    mqi::vec3<T> dir(std::atan(x/SAD_[0]), std::atan(y/SAD_[1]), -1.0);

        std::array<T,6> phsp ; 
        T Ux = func_(*rng); T Vx = func_(*rng);
        T Uy = func_(*rng); T Vy = func_(*rng);

        phsp[0] = x + pdf_Md<T,6>::sigma_[0]* Ux ; 
        phsp[1] = y + pdf_Md<T,6>::sigma_[1]* Uy ; 
        phsp[2] = pdf_Md<T,6>::mean_[5] ;  

        phsp[3] = dir.x + pdf_Md<T,6>::sigma_[3] * (rho_[0] * Ux + Vx * std::sqrt(1.0-rho_[0]*rho_[0])); 
        phsp[4] = dir.y + pdf_Md<T,6>::sigma_[4] * (rho_[1] * Uy + Vy * std::sqrt(1.0-rho_[1]*rho_[1]));
        phsp[5] =  -1.0*std::sqrt( 1.0 - phsp[3]*phsp[3]- phsp[4]*phsp[4] );

        return phsp;
    };

};



}

#endif

