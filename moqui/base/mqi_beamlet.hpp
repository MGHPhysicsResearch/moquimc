#ifndef MQI_BEAMLET_HPP
#define MQI_BEAMLET_HPP

/// \file
///
/// A beamlet is a collection of distributions of a beam model and
/// provides phase-space variables, e, x, x', y, y', z, z'.

#include <array>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <tuple>

#include <moqui/base/mqi_coordinate_transform.hpp>
#include <moqui/base/mqi_distributions.hpp>
#include <moqui/base/mqi_vertex.hpp>

namespace mqi
{

/// \class beamlet
/// Represents a beamlet of a treatment field.
/// Beamlet can be a a whole field (maybe double scattering) or a pencil beam of IMPT
/// Beamlet consists of 4 distributions
/// 1. energy samples energy cooof beam
/// 2. fluence samples x, x', y, y', z, z'
/// 3. mapping coordinate system
/// \tparam T for types for return values by the distributions
/// \note
/// This beamlet may be called CPU and GPU.
template<typename T>
class beamlet
{
protected:
    /// Distribution function of energy, 1-D distribution
    mqi::pdf_Md<T, 1>* energy = nullptr;

    /// Distribution function for fluence, 6-D distribution
    mqi::pdf_Md<T, 6>* fluence = nullptr;

    /// Coordinate transform to map local generation to patient or treatment coordination.
    coordinate_transform<T> p_coord;

public:
    /// Construct a beam from the distributions of energy and fluence.
    CUDA_HOST_DEVICE
    beamlet(mqi::pdf_Md<T, 1>* e, mqi::pdf_Md<T, 6>* f) : energy(e), fluence(f) {
        ;
    }

    /// Destructor
    /// \note Do we need to de-allocate the energy and fluence distribution here?
    CUDA_HOST_DEVICE
    beamlet() {
        ;
    }

    /// Creates a copy beamlet from assignment operator
    CUDA_HOST_DEVICE
    beamlet(const beamlet<T>& rhs) {
        energy  = rhs.energy;
        fluence = rhs.fluence;
        p_coord = rhs.p_coord;
    }

    /// Set coordinate transform from outside.
    /// \param p coordinate_transform consisting of translation and rotation
    CUDA_HOST_DEVICE
    void
    set_coordinate_transform(coordinate_transform<T> p) {
        p_coord = p;
    }

    /// Samples energy, position, direction of a history
    /// \param no
    /// \return a tuple of energy, position(vec3), direction (vec3)
    //    CUDA_HOST_DEVICE
    virtual mqi::vertex_t<T>
    operator()(std::default_random_engine* rng) {
        std::array<T, 6> phsp = (*fluence)(rng);
        mqi::vec3<T>     pos(phsp[0], phsp[1], phsp[2]);
        mqi::vec3<T>     dir(phsp[3], phsp[4], phsp[5]);
        mqi::vertex_t<T> vtx;
        vtx.ke  = (*energy)(rng)[0];
        vtx.pos = p_coord.rotation * pos + p_coord.translation;
        vtx.dir = p_coord.rotation * dir;
        return vtx;
    };
};

}   // namespace mqi

#endif
