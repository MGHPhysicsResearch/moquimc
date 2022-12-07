#ifndef MQI_COORDINATE_TRANSFORM_HPP
#define MQI_COORDINATE_TRANSFORM_HPP

/// \file
///
/// A coordinate transform to map a history from beamsource to map a IEC coordinate or DICOM coordinate.

#include <array>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <tuple>

#include <moqui/base/mqi_matrix.hpp>
#include <moqui/base/mqi_vec.hpp>

namespace mqi
{

/// \class coordinate_transform
/// default units are mm for length and degree for angle.
/// rotation direction is CCW
/// order of rotation is collimator->gantry->couch->iec2dicom
/// \tparam T for types for return values by the distributions
template<typename T>
class coordinate_transform
{
public:
    /// A constant to convert degree to radian.
    const float deg2rad = M_PI / 180.0;

    ///< translation
    vec3<T> translation;

    ///< rotation
    std::array<T, 4> angles { 0.0, 0.0, 0.0, 0.0 };

    /// rotation matrices
    mat3x3<T> collimator;        ///< Rotation due to collimator angle
    mat3x3<T> gantry;            ///< Rotation due to gantry angle
    mat3x3<T> patient_support;   ///< Rotation due to couch
    mat3x3<T> iec2dicom;         ///-90 deg (iec2dicom), 90 deg (dicom2iec)

    /// Final rotation matrix and translation vector
    mat3x3<T> rotation;

    /// Constructor with 4-angles and position
    /// \param angles[4] angles of collimator, gantry, couch, and iec
    /// \param pos move origin, i.e, isocenter
    CUDA_HOST_DEVICE
    coordinate_transform(std::array<T, 4>& ang, vec3<T>& pos) : angles(ang), translation(pos) {
        collimator      = mat3x3<T>(0, 0, angles[0] * deg2rad);
        gantry          = mat3x3<T>(0, angles[1] * deg2rad, 0);
        patient_support = mat3x3<T>(0, 0, angles[2] * deg2rad);
        iec2dicom       = mat3x3<T>(angles[3] * deg2rad, 0, 0);
        rotation        = iec2dicom * patient_support * gantry * collimator;
    }

    /// Constructor with constant 4-angles and position
    /// \param const angles[4] angles of collimator, gantry, couch, and iec
    /// \param const pos
    CUDA_HOST_DEVICE
    coordinate_transform(const std::array<T, 4>& ang, const vec3<T>& pos) :
        angles(ang), translation(pos) {
        collimator      = mat3x3<T>(0, 0, angles[0] * deg2rad);
        gantry          = mat3x3<T>(0, angles[1] * deg2rad, 0);
        patient_support = mat3x3<T>(0, 0, angles[2] * deg2rad);
        iec2dicom       = mat3x3<T>(angles[3] * deg2rad, 0, 0);
        rotation        = iec2dicom * patient_support * gantry * collimator;
    }

    /// A copy constructor
    CUDA_HOST_DEVICE
    coordinate_transform(const coordinate_transform<T>& ref) : angles(ref.angles) {
        collimator      = ref.collimator;
        gantry          = ref.gantry;
        patient_support = ref.patient_support;
        iec2dicom       = ref.iec2dicom;
        rotation        = iec2dicom * patient_support * gantry * collimator;
        translation     = ref.translation;
    }

    /// Destructor
    CUDA_HOST_DEVICE
    coordinate_transform() : angles() {
        ;
    }

    /// Assignment operator
    CUDA_HOST_DEVICE
    coordinate_transform<T>&
    operator=(const coordinate_transform<T>& ref) {
        collimator      = ref.collimator;
        gantry          = ref.gantry;
        patient_support = ref.patient_support;
        iec2dicom       = ref.iec2dicom;
        rotation        = iec2dicom * patient_support * gantry * collimator;
        translation     = ref.translation;
        angles          = ref.angles;
        return *this;
    }

    /// Prints out matrix
    CUDA_HOST
    void
    dump() {
        std::cout << "--- coordinate transform---" << std::endl;
        std::cout << "    translation ---" << std::endl;
        translation.dump();
        std::cout << "    rotation ---" << std::endl;
        rotation.dump();
        std::cout << "    gantry ---" << std::endl;
        gantry.dump();
        std::cout << "    patient_support ---" << std::endl;
        patient_support.dump();
        std::cout << "    collimator ---" << std::endl;
        collimator.dump();
        std::cout << "    IEC2DICOM ---" << std::endl;
        iec2dicom.dump();
    }
};

}   // namespace mqi

#endif
