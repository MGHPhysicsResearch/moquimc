#ifndef MQI_GEOMETRY_HPP
#define MQI_GEOMETRY_HPP

/// \file
///
/// RT-Ion geometry

#include <array>
#include <map>
#include <string>
#include <vector>

#include <moqui/base/mqi_matrix.hpp>
#include <moqui/base/mqi_vec.hpp>

namespace mqi
{

/// Enumerate for geometry type
/// \note many of these are not supported yet.
typedef enum
{
    SNOUT,
    RANGESHIFTER,
    COMPENSATOR,
    BLOCK,
    BOLI,
    WEDGE,
    TRANSFORM,
    MLC,
    PATIENT,
    DOSEGRID,
    UNKNOWN1,
    UNKNOWN2,
    UNKNOWN3,
    UNKNOWN4
} geometry_type;

/// \class geometry
/// Top abstraction class so it has only translation and rotational movement.
class geometry
{

public:
    const mqi::vec3<float>   pos;       ///< position, public access
    const mqi::mat3x3<float> rot;       ///< rotation, public access
    const geometry_type      geotype;   ///< geometry type, public access

    /// Constructor
    /// \param p_xyz position vector
    /// \param rot_xyz rotation matrix 3x3
    geometry(mqi::vec3<float>& p_xyz, mqi::mat3x3<float>& rot_xyz, mqi::geometry_type t) :
        pos(p_xyz), rot(rot_xyz), geotype(t) {
        ;
    }

    /// Constructor
    /// \param p_xyz position vector
    /// \param rot_xyz rotation matrix 3x3
    geometry(const mqi::vec3<float>&   p_xyz,
             const mqi::mat3x3<float>& rot_xyz,
             const mqi::geometry_type  t) :
        pos(p_xyz),
        rot(rot_xyz), geotype(t) {
        ;
    }

    /// Copy constructor
    geometry(const geometry& rhs) : geotype(rhs.geotype), pos(rhs.pos), rot(rhs.rot) {
        ;
    }

    /// Assignment operator
    const geometry&
    operator=(const mqi::geometry& rhs) {
        return rhs;
    }

    /// Destructor
    ~geometry() {
        ;
    }

    /// Prints geometry information
    virtual void
    dump() const {
        ;
    }
};

}   // namespace mqi
#endif
