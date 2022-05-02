#ifndef MQI_TREATMENT_MACHINE_H
#define MQI_TREATMENT_MACHINE_H

/// \file
///
/// Abstraction for treatment machine

#include <moqui/base/mqi_beam_module.hpp>
#include <moqui/base/mqi_beam_module_ion.hpp>
#include <moqui/base/mqi_beamlet.hpp>
#include <moqui/base/mqi_beamline.hpp>
#include <moqui/base/mqi_beamsource.hpp>

namespace mqi
{

/// \class treatment_machine
///
/// Describes an abstraction of all types (RT and ION) treatment machine
/// Typically it consists of geometries (beam limiting devices) and sources (particle fluence)
/// This class has three user-implemented methods, create_beamline, create_beamsource, and create_coordinate_transform
/// \tparam T type of phase-space variables, e.g., float or double
/// \note treatment machine is distinguished in its name
template<typename T>
class treatment_machine
{
protected:
    ///< Machine name in string (site:system:mc_code)
    const std::string name_;

    ///< Source to Axis distance,
    ///< neccessary to calculate beam divergence
    std::array<float, 2> SAD_;

    ///< Distance from phase-space plan to isocenter
    float source_to_isocenter_mm_;

public:
    /// Default constructor
    treatment_machine() {
        ;
    }

    /// Virtual destructor of abstract base class
    virtual ~treatment_machine() {
        ;
    }

    /// Returns beamline model
    /// \param ds : pointer of dataset
    /// \param m  : modality type such as RTIP, US, PASSIVE, etc
    /// \return beamline
    virtual mqi::beamline<T>
    create_beamline(const mqi::dataset* ds, mqi::modality_type m) = 0;

    /// Returns beamsource model
    /// \param ds : pointer of dataset
    /// \param m  : modality type such as RT-Ion Plan, RT-Ion Beam Treatment Record
    /// \param pcoord: coordinate system of beam geometry including gantry, patient support, etc
    /// \param particles_per_history : total number of histories can be calculated using this number
    virtual mqi::beamsource<T>
    create_beamsource(const mqi::dataset*                ds,
                      const mqi::modality_type           m,
                      const mqi::coordinate_transform<T> pcoord,
                      const float                        particles_per_history  = -1,
                      const float                        source_to_isocenter_mm = 390.0) = 0;

    /// Returns beamsource model from file, e.g., tramp for MGH
    /// \param pcoord: coordinate system of beam geometry including gantry, patient support, etc
    /// \param particles_per_history : total number of histories can be calculated using this number
    virtual mqi::beamsource<T>
    create_beamsource(const std::vector<mqi::beam_module_ion::spot>& spots,
                      const mqi::modality_type                       m,
                      const mqi::coordinate_transform<T>             pcoord,
                      const float                                    particles_per_history = -1,
                      const float source_to_isocenter_mm = 390.0) = 0;

    /// Returns coordinate transform information
    /// \param ds : pointer of dataset
    /// \param m  : modality type such as IMPT, US, PASSIVE, etc
    /// \return pcoord: coordinate system of beam geometry including gantry, patient support, etc
    virtual mqi::coordinate_transform<T>
    create_coordinate_transform(const mqi::dataset* ds, const mqi::modality_type m) = 0;

protected:
    /// Returns beamlet
    /// \param s : spot contains energy, x-, and y-position, fwhm, and NP
    /// \return beamlet: beamlet created from given parameters
    virtual mqi::beamlet<T>
    characterize_beamlet(const mqi::beam_module_ion::spot& s) = 0;

    /// Returns beamlet
    /// \param s : spot contains energy, x-, and y-position, fwhm, and NP
    /// \param source_to_isocenter_mm : distance between source and isocenter in mm scale
    /// \return beamlet: beamlet created from given parameters
    virtual mqi::beamlet<T>
    characterize_beamlet(const mqi::beam_module_ion::spot& s,
                         const float                       source_to_isocenter_mm) = 0;
};

}   // namespace mqi

#endif
