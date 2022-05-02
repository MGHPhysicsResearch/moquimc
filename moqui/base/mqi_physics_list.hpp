#ifndef MQI_PHYSICS_LIST_HPP
#define MQI_PHYSICS_LIST_HPP

#include <moqui/base/mqi_interaction.hpp>

namespace mqi
{

///< struct for Minimum step length and process id
//P-> 0: CSDA, 1: delta_ion, 2: pp-elastic, 3: po-elastic, 4: po-inelastic
//P-> -1: by geometry
template<typename R>
struct dL_t {
    R      L;
    int8_t P;
};

///<
template<typename R>
class physics_list
{
public:
    const physics_constants<R> units;
    R       Te_cut   = 0.08511 * units.MeV;   //0.1 will set mpf 0 up to 45 MeV proton
    const R Tp_cut   = 0.5 * units.MeV;
    const R Tp_max   = 330.0 * units.MeV;   // maximum proton energy to deal
    const R Tp_up    = 2.0 * units.MeV;     //restricted stopping power
    const R max_step = 1.0 * units.cm;

    ///Number of processes per particle
    ///< photon, electron, proton, neutron
public:
    CUDA_HOST_DEVICE
    physics_list() {
        ;
    }

    CUDA_HOST_DEVICE
    ~physics_list() {}
};

}   // namespace mqi
#endif
