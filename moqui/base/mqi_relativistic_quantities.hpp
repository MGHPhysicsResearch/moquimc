#ifndef MQI_REL_QUANTITIES_HPP
#define MQI_REL_QUANTITIES_HPP

#include <moqui/base/mqi_common.hpp>
#include <moqui/base/mqi_math.hpp>

namespace mqi
{

template<typename R>
class relativistic_quantities
{
public:
    R beta_sq;    //= beta2_p(ke) ;
    R gamma_sq;   //= gamma2_p(ke);
    R gamma;      //= gamma_p(ke) ;
    R Et;         //= Ek + mass
    R Et_sq;      //Et * Et
    R Ek;         //= kinetic energy
    R mc2;        //m0c^2: rest mass
    R tau;        // Ek/Mp for energy loss approximation by Kwarakw, 2000.
    R Te_max;     //maximum transfer energy to electron by proton (will extend to other particles)

    CUDA_HOST_DEVICE
    relativistic_quantities(R kinetic_energy, R rest_mass_MeV) :
        Ek(kinetic_energy), mc2(rest_mass_MeV) {
        const R Mp      = 938.272046;   // Mp/eV = proton mass in eV
        const R Me      = 0.510998928;
        Et              = Ek + Mp;
        Et_sq           = Et * Et;
        const R MeMp    = Me / Mp;       // Me/Mp
        const R MeMp_sq = MeMp * MeMp;   // (Me/Mp)^2
        gamma           = Et / Mp;
        gamma_sq        = gamma * gamma;
        beta_sq         = 1.0 - 1.0 / gamma_sq;

        Te_max = (2.0 * Me * beta_sq * gamma_sq);
        Te_max /= (1.0 + 2.0 * gamma * MeMp + MeMp_sq);
        tau = Ek / Mp;
    }

    CUDA_HOST_DEVICE
    ~relativistic_quantities() {
        ;
    }

    CUDA_HOST_DEVICE
    R
    momentum() {
        return mqi::mqi_sqrt(Et * Et - mc2 * mc2);
    }
};

}   // namespace mqi
#endif
