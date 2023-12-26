#ifndef MQI_INTERACTION_HPP
#define MQI_INTERACTION_HPP
#include <random>

#include <moqui/base/mqi_material.hpp>
#include <moqui/base/mqi_math.hpp>
#include <moqui/base/mqi_physics_constants.hpp>
#include <moqui/base/mqi_relativistic_quantities.hpp>
#include <moqui/base/mqi_track.hpp>
#include <moqui/base/mqi_track_stack.hpp>

namespace mqi
{

///< Interaction model (pure virtual class)
///< interface between particle and material
///< template R and particle type
///<
template<typename R, mqi::particle_t P>
class interaction
{
public:
    const physics_constants<R> units;
#ifdef __PHYSICS_DEBUG__
    R T_cut = 0.08511 * units.MeV;   //0.083 MeV for electron cut, 0.01 MeV cut for Proton
#else
    R T_cut = 0.0815 * units.MeV;   //0.083 MeV for electron cut, 0.01 MeV cut for Proton
#endif
    const R max_step = 0.01 * units.cm;   //0.1 * mm cut used to generate dEdx, cross-section
    R       Tp_cut   = 0.5 * units.MeV;
    const mqi::vec3<R> dir_z;   //momentum direction of incident particle on scattering plane

public:
    CUDA_HOST_DEVICE
    interaction() : dir_z(0, 0, -1) {
        ;
    }

    CUDA_HOST_DEVICE
    ~interaction() {
        ;
    }

//    ///< sample step length (cm)
//    /// rho_mass (cm^3)
//    CUDA_HOST_DEVICE
//    virtual R
//    sample_step_length(const relativistic_quantities<R>& rel,
//                       const material_t<R>&              mat,
//                       mqi_rng*                          rng) {
//        R cs   = mat.rho_mass * this->cross_section(rel, mat);
//        R mfp  = (cs == 0.0) ? max_step : 1.0 / cs;
//        R prob = mqi_uniform<R>(rng);
//        return -1.0 * mfp * mqi_ln(prob);
//    }
//
//    ///< sample step length (cm)
//    /// rho_mass (cm^3)
//    CUDA_HOST_DEVICE
//    virtual R
//    sample_step_length(const R cs, mqi_rng* rng) {
//        R mfp  = (cs == 0.0) ? max_step : 1.0 / cs;
//        R prob = mqi_uniform<R>(rng);
//        return -1.0 * mfp * mqi_ln(prob);
//    }

    ///< Return cross-section of the process in mm unit
    ///< pure virtual method so that a child needs to fill
    CUDA_HOST_DEVICE
    virtual R
    cross_section(const relativistic_quantities<R>& rel, material_t<R>*& mat, R rho_mass) = 0;

    ///< Update track during a given step
    ///< e.g., this is usually for CSDA energy loss
    CUDA_HOST_DEVICE
    virtual void
    along_step(track_t<R>&       trk,
               track_stack_t<R>& stk,
               mqi_rng*          rng,
               const R           len,
               material_t<R>*&   mat,
               R                 rho_mass) = 0;

    ///< Update track at the end of step
    ///< e.g., push secondaries to the stack
    ///        update track status, e.g., created, stopped
    CUDA_HOST_DEVICE
    virtual void
    post_step(track_t<R>&       trk,
              track_stack_t<R>& stk,
              mqi_rng*          rng,
              const R           len,
              material_t<R>*&   mat,
              bool              score_local_deposit) = 0;
};

}   // namespace mqi

#endif
