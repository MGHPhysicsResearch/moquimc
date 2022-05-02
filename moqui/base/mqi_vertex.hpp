#ifndef MQI_VERTEX_HPP
#define MQI_VERTEX_HPP

#include <moqui/base/mqi_vec.hpp>

namespace mqi
{

///< Particle properties at a position
///< Physics will propose next vertex
///< Geometry will propose next vertex
///< Step limiter will propose next vertex
///< Magnetic field will propose next vertex
///< Vertex doesn't include particle type
template<typename T>
struct vertex_t {
    T       ke;    //< kinetic energy
    vec3<T> pos;   //< position
    vec3<T> dir;   //< direction

    CUDA_HOST_DEVICE
    vertex_t<T>&
    operator=(const vertex_t<T>& rhs) {
        ke  = rhs.ke;
        pos = rhs.pos;
        dir = rhs.dir;
        return *this;
    }
};

}   // namespace mqi
#endif
