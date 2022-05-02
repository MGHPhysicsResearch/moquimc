#ifndef MQI_BEAMLINE_H
#define MQI_BEAMLINE_H

/// \file
///
/// A beamline is a collection of physical components in the beamline

#include <moqui/base/mqi_coordinate_transform.hpp>
#include <moqui/base/mqi_geometry.hpp>

namespace mqi
{

/// \class beamline
///
/// \tparam T type of units
/// \note we may not need this class if this is a just container for geometries.
/// Will be determined later.
template<typename T>
class beamline
{
public:
protected:
    /// A container for components
    std::vector<mqi::geometry*> geometries_;

public:
    /// An empty constructor
    beamline() {
        ;
    }

    /// Clear all geometries in the vector
    ~beamline() {
        geometries_.clear();
    }

    /// Add new geometry to the container.
    void
    append_geometry(mqi::geometry* geo) {
        geometries_.push_back(geo);
    }

    /// Returns geometry container (const reference)
    const std::vector<mqi::geometry*>&
    get_geometries() {
        return geometries_;
    }
};

}   // namespace mqi
#endif
