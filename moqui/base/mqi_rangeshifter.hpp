#ifndef MQI_RANGESHIFTER_H
#define MQI_RANGESHIFTER_H

/// \file
///
/// RT-Ion geometry for rangeshifter

#include <moqui/base/mqi_geometry.hpp>

namespace mqi
{

/// \class rangeshifter
/// supports box and cylinder type rangeshifter
/// \note only one material
class rangeshifter : public geometry
{

public:
    const bool             is_rectangle;   ///< type of volume
    const mqi::vec3<float> volume;         ///< volume dimension

    /// Constructor
    rangeshifter(mqi::vec3<float>&   v,   ///< x,y,z or r, theta, thickness
                 mqi::vec3<float>&   p,   ///< position
                 mqi::mat3x3<float>& r,   ///< rotation matrix
                 bool                is_rect = true) :
        volume(v),
        is_rectangle(is_rect), geometry(p, r, mqi::geometry_type::RANGESHIFTER) {
        ;
    }

    /// Copy constructor
    rangeshifter(const mqi::rangeshifter& rhs) :
        volume(rhs.volume), is_rectangle(rhs.is_rectangle),
        geometry(rhs.pos, rhs.rot, rhs.geotype) {
        ;
    }

    /// Destructor
    ~rangeshifter() {
        ;
    }

    /// Assignment operator
    const rangeshifter&
    operator=(const mqi::rangeshifter& rhs) {
        return rhs;
    }
};

}   // namespace mqi

#endif