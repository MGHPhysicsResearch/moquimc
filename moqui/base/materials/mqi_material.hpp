#ifndef MQI_MATERIAL_HPP
#define MQI_MATERIAL_HPP

#include <moqui/base/mqi_common.hpp>
#include <moqui/base/mqi_math.hpp>
#include <moqui/base/mqi_physics_constants.hpp>

namespace mqi
{
template<typename R>
using fp_compute_rsp = float (*)(R rho_mass, R Ek);
template<typename R>
using fp_compute_rl = R (*)(R rho_mass, R water_density, R radiation_length_water);
typedef uint16_t material_id;
///< Interaction model (pure virtual class)
///< interface between particle and material
///< use template R for density type
///<
template<typename R>
class material_t
{
public:
    physics_constants<R> units;   //TODO: better to be defined as global variable.
    fp_compute_rsp<R>    compute_rsp_;
    fp_compute_rl<R>     compute_rl_;

public:
    CUDA_HOST_DEVICE
    material_t() {
        ;
    }

    CUDA_HOST_DEVICE
    ~material_t() {
        ;
    }

    CUDA_HOST_DEVICE
    material_t<R>&
    operator=(const material_t<R>& r) {
        return *this;
    }

    CUDA_HOST_DEVICE
    inline R
    dedx_term0() const {
        return this->units.two_pi_re2_mc2_h2o;
    }
};
}   // namespace mqi

#endif
