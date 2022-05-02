#ifndef MQI_APERTURE_H
#define MQI_APERTURE_H

/// \file
///
/// Geometry model for an aperture

#include <moqui/base/mqi_geometry.hpp>

namespace mqi
{

/// \class aperture
/// \brief A aperture class

class aperture : public geometry
{
public:
    /// whehter aperture shape is box or cylinder type
    const bool is_rectangle;

    /// aperture dimension, (Lx, Ly, Lz) for box or (R, H, dummy) for cylinder
    const mqi::vec3<float> volume;

    /// X-Y opening points. Divergence is not considered.
    const std::vector<std::vector<std::array<float, 2>>> block_data;

public:
    /// Creates an aperture
    /// \param xypts a set of (x,y) points.
    /// \param v a volume, e.g., (Lx,Ly,Lz) or (R, thickness, ignored).
    /// \param p a center position of the aperture.
    /// \param is_rect tells whether aperture is box shape or cylinder shape.
    aperture(std::vector<std::vector<std::array<float, 2>>> xypts,
             mqi::vec3<float>&                              v,
             mqi::vec3<float>&                              p,
             mqi::mat3x3<float>&                            r,
             bool                                           is_rect = true) :
        block_data(xypts),
        volume(v), is_rectangle(is_rect), geometry(p, r, mqi::geometry_type::BLOCK) {
        ;
    }

    /// Creates a copy from an existing aperture
    aperture(const mqi::aperture& rhs) :
        volume(rhs.volume), block_data(rhs.block_data), is_rectangle(rhs.is_rectangle),
        geometry(rhs.pos, rhs.rot, rhs.geotype) {
        ;
    }

    /// Destructor
    ~aperture() {
        ;
    }

    /// Assignment operator
    const aperture&
    operator=(const mqi::aperture& rhs) {
        return rhs;
    }
};

}   // namespace mqi
#endif