#ifndef MQI_TREATMENT_MACHINE_ION_H
#define MQI_TREATMENT_MACHINE_ION_H

/// \file
///
/// Treatment machine for particle therapy

#include <moqui/base/mqi_aperture.hpp>
#include <moqui/base/mqi_beam_module_ion.hpp>
#include <moqui/base/mqi_rangeshifter.hpp>
#include <moqui/base/mqi_treatment_machine.hpp>

#define SIGMA2FWHM 2.35482004503   // 2.*std::sqrt(2.*std::log(2.))

namespace mqi
{

/// \class treatment_machine_ion
///
/// Describes an particle therapy system
/// \tparam T type of phase-space variables, e.g., float or double
template<typename T>
class treatment_machine_ion : public treatment_machine<T>
{

protected:
public:
    /// Default constructor
    treatment_machine_ion() {
        ;
    }

    /// Default Destructors
    ~treatment_machine_ion() {
        ;
    }

    /// Returns angles of Collimator, Gantry, and Couch and iso-center position
    /// \note rotation angle directio is assumed CCW
    /// As RTIBTR doesn't have iso-center position, (0,0,0) is returned for RTIBTR
    virtual mqi::coordinate_transform<T>
    create_coordinate_transform(const mqi::dataset* ds, const mqi::modality_type m) {

        std::cout << "create_coordinate_transform:" << std::endl;
        auto                     seq_tags = &mqi::seqtags_per_modality.at(m);
        auto                     layer0   = (*ds)(seq_tags->at("ctrl"))[0];   //layer0
        std::vector<float>       tmp;
        std::vector<std::string> tmp_dir;
        std::array<T, 4>         angles;
        ///< As CW, CRW, NONE, conditionally required, we don't use them at this moment
        layer0->get_values("BeamLimitingDeviceAngle", tmp);
        angles[0] = tmp[0];
        layer0->get_values("GantryAngle", tmp);
        angles[1] = tmp[0];

        layer0->get_values("PatientSupportAngle", tmp);
        angles[2] = -1.0 * tmp[0];
        angles[3] = 0.0;

        mqi::vec3<T> pos;

        ///< Rotation and Position don't exist in RTRECORD and IONRECORD
        if ((m == RTPLAN) || (m == IONPLAN)) {
            layer0->get_values("IsocenterPosition", tmp);
            pos.x = tmp[0];
            pos.y = tmp[1];
            pos.z = tmp[2];
        }
        return mqi::coordinate_transform<T>(angles, pos);
    }

    /// Returns beamsource object
    /// \param ds one-item of IonBeamSequence
    /// \param m  modality type, e.g., RTIP or RTIBTR
    /// \param pcoord  coordinate transform to map global coordinate system
    /// \param scalefactor a downscale factor, i.e., particles per history.
    ///        -1 means any of beamlet generates one history (usefule for debugging).
    /// \param source_to_isocenter_mm distance the histories are generated.
    /// \note user don't have to touch this part typically.
    /// \note currently we support only MODULATED beam
    /// \note maybe return const mqi::beamsource<T>& ?
    ///
    virtual mqi::beamsource<T>
    create_beamsource(const mqi::dataset*                ds,
                      const mqi::modality_type           m,
                      const mqi::coordinate_transform<T> pcoord,
                      const float                        particles_per_history  = -1,
                      const float                        source_to_isocenter_mm = 390.0) {
        treatment_machine<T>::source_to_isocenter_mm_ = source_to_isocenter_mm;

        ///< Parse DICOM beam module for Ion
        beam_module_ion ion_beam(ds, m);

        std::vector<std::string> scan_mode(1);
        ds->get_values("ScanMode", scan_mode);
        if (!scan_mode[0].compare("MODULATED")) {
            std::runtime_error("Only MODULATED scan mode is supported");
        }
        mqi::beamsource<T> beamsource;

        ///< Modulated BEAM
        const auto&                       spots     = *(ion_beam.get_sequence());
        const size_t                      nb_spots  = spots.size();
        const mqi::beam_module_ion::spot* null_spot = nullptr;

        for (size_t i = 0; i < nb_spots; ++i) {
            /// Calculate number of histories to be simulated per beamlet
            size_t nb_histories = (particles_per_history == -1)
                                    ? 1
                                    : this->characterize_history(spots[i], particles_per_history);

            /// Calculate on & off time per beamlet
            /// By default, on is set to 1 sec but off is set to 0 sec.

            std::array<T, 2> time_on_off;
            if (i == (nb_spots - 1)) {
                time_on_off = this->characterize_beamlet_time(spots[i], *null_spot);

            } else {
                time_on_off = this->characterize_beamlet_time(spots[i], spots[i + 1]);
            }
            /// Then, add a beamlet with
            ///            its number of histories,
            ///            coordinate system
            ///            beamlet time on/off
            beamsource.append_beamlet(this->characterize_beamlet(spots[i]),
                                      nb_histories,
                                      pcoord,
                                      time_on_off[0],
                                      time_on_off[1]);
        }

        return beamsource;
    }

    /// Returns beamsource object
    /// \param ds one-item of IonBeamSequence
    /// \param m  modality type, e.g., RTIP or RTIBTR
    /// \param pcoord  coordinate transform to map global coordinate system
    /// \param particles_per_history a downscale factor, i.e., particles per history.
    ///        -1 means any of beamlet generates one history (usefule for debugging).
    /// \param source_to_isocenter_mm distance the histories are generated.
    /// \note user don't have to touch this part typically.
    /// \note currently we support only MODULATED beam
    /// \note maybe return const mqi::beamsource<T>& ?
    ///
    virtual mqi::beamsource<T>
    create_beamsource(const std::vector<mqi::beam_module_ion::spot>& spots,
                      const mqi::modality_type                       m,
                      const mqi::coordinate_transform<T>             pcoord,
                      const float                                    particles_per_history = -1,
                      const float source_to_isocenter_mm = 390.0) {

        treatment_machine<T>::source_to_isocenter_mm_ = source_to_isocenter_mm;
        ///< Parse DICOM beam module for ION
        mqi::beamsource<T> beamsource;

        ///< Modulated BEAM
        const size_t                      nb_spots  = spots.size();
        const mqi::beam_module_ion::spot* null_spot = nullptr;
        for (size_t i = 0; i < nb_spots; ++i) {
            /// Calculate number of histories to be simulated per beamlet

            //// TODO: history counting is different from the other method and would be very confisuing
            size_t nb_histories = (particles_per_history <= 1)
                                    ? 1
                                    : std::ceil(spots[i].meterset * particles_per_history);

            /// Calculate on & off time per beamlet
            /// By default, on is set to 1 sec but off is set to 0 sec.
            std::array<T, 2> time_on_off;
            if (i == (nb_spots - 1)) {
                time_on_off = this->characterize_beamlet_time(spots[i], *null_spot);
            } else {
                time_on_off = this->characterize_beamlet_time(spots[i], spots[i + 1]);
            }

            /// Then, add a beamlet with
            ///            its number of histories,
            ///            coordinate system
            ///            beamlet time on/off
            beamsource.append_beamlet(this->characterize_beamlet(spots[i]),
                                      nb_histories,
                                      pcoord,
                                      time_on_off[0],
                                      time_on_off[1]);
        }
        return beamsource;
    }

    /// User method to characterize MODULATED beamlet based on spot information from DICOM.
    virtual mqi::beamlet<T>
    characterize_beamlet(const mqi::beam_module_ion::spot& s) = 0;

    /// User method to characterize MODULATED beamlet based on spot information ans source to isocenter distnace from DICOM.
    virtual mqi::beamlet<T>
    characterize_beamlet(const mqi::beam_module_ion::spot& s,
                         const float                       source_to_isocenter_mm) = 0;

    /// User method to characterize UNIFORM/MODULATED_SPEC beamlet based on spot information from DICOM.
    virtual mqi::beamlet<T>
    characterize_beamlet(const mqi::beam_module_ion::spot& s0,
                         const mqi::beam_module_ion::spot& s1) = 0;

    /// User method to characterize beam delivery time
    /// on_time, off_time by default 1 sec and 0 sec
    virtual std::array<T, 2>
    characterize_beamlet_time(const mqi::beam_module_ion::spot& s_current,
                              const mqi::beam_module_ion::spot& s_next) {
        return { 1.0, 0.0 };
    }

    /// User method to characterize number of histories per MODULATED beamlet
    /// based on spot information from DICOM.
    virtual size_t
    characterize_history(const mqi::beam_module_ion::spot& s, float scale) {
        return s.meterset / scale;
    }

    /// User method to characterize number of histories per UNIFORM/MODULATED_SPEC beamlet
    /// based on spot information from DICOM.
    virtual size_t
    characterize_history(const mqi::beam_module_ion::spot& s0,
                         const mqi::beam_module_ion::spot& s1,
                         float                             scale) {
        return s1.meterset / scale;
    }

    /// Returns beamline object
    /// \param ds one-item of IonBeamSequence
    /// \param m  modality type, e.g., RTIP or RTIBTR
    /// \note Users don't have to reimplement this method typically.
    /// \note maybe return const mqi::beamline<T>& ?
    /// \note coordinate_transform is not implemented here
    /// \note coordinate_transform is handled in MC engine, i.e., TOPAS by now.
    /// \note it's not yet clear whether to include coordinate_transform like create_beamsource
    /// \note currently we consider range-shiter and aperture. geometry will be added later.
    virtual mqi::beamline<T>
    create_beamline(const mqi::dataset* ds, mqi::modality_type m) {
        mqi::beamline<T> beamline;

        ///< Access to tag LUT
        auto seq_tags = &mqi::seqtags_per_modality.at(m);

        ///< geometry creation of snout?
        ///< position from control point 0
        ///< beamline.append_geometry(this->characterize_snout);
        std::vector<int> itmp;
        std::vector<int> ftmp;

        ///< 1. number of rangeshifter sequence
        ds->get_values("NumberOfRangeShifters", itmp);
        std::cout << "number of range shifter: " << itmp[0] << std::endl;
        if (itmp[0] >= 1) { beamline.append_geometry(this->characterize_rangeshifter(ds, m)); }

        ///< 2. number of blocks
        ds->get_values("NumberOfBlocks", itmp);
        std::cout << "number of blocks: " << itmp[0] << std::endl;
        if (itmp[0] >= 1) { beamline.append_geometry(this->characterize_aperture(ds, m)); }
        return beamline;
    }

    /// Returns rangeshifter
    /// \param ds one-item of IonBeamSequence
    /// \param m  modality type, e.g., RTIP or RTIBTR
    /// \note Users need to implement their own rule to convert DICOM info to RS specification.
    virtual mqi::rangeshifter*
    characterize_rangeshifter(const mqi::dataset* ds, mqi::modality_type m) = 0;

    /// Returns aperture
    /// \param ds one-item of IonBeamSequence
    /// \param m  modality type, e.g., RTIP or RTIBTR
    /// \note Users need to implement their own rule to convert DICOM info to RS specification.
    virtual mqi::aperture*
    characterize_aperture(const mqi::dataset* ds, mqi::modality_type m) = 0;

    /// Returns set of (x,y) points for aperture openning.
    /// \param ds one-item of IonBeamSequence
    /// \param m  modality type, e.g., RTIP or RTIBTR
    /// \note Users need to implement their own rule to convert DICOM info to APT specification.
    /// \note multiple holes are supported.
    const std::vector<std::vector<std::array<float, 2>>>
    characterize_aperture_opening(const mqi::dataset* ds, mqi::modality_type m) {
        std::vector<std::vector<std::array<float, 2>>> apt_xy_points;
        auto                                           seq_tags = &mqi::seqtags_per_modality.at(m);

        //0. aperture sequence
        auto apt_ds = (*ds)(seq_tags->at("blk"));
        assert(apt_ds.size() >= 1);

        for (auto apt : apt_ds) {
            std::vector<std::array<float, 2>> block_data;
            std::vector<int>                  nb_xy_points;
            std::vector<float>                xy_points;

            apt->get_values("BlockNumberOfPoints", nb_xy_points);
            apt->get_values("BlockData", xy_points);

            for (int j = 0; j < nb_xy_points[0]; ++j) {
                block_data.push_back({ xy_points[j * 2], xy_points[j * 2 + 1] });
            }
            apt_xy_points.push_back(block_data);
        }
        return apt_xy_points;
    }

    /// Returns set of (x,y) points at z-position.
    /// \param iso : isocenter position. iso.z is not read in by default.
    /// \param z   : z-position where the beam starts
    virtual mqi::vec3<T>
    beam_starting_position(const mqi::vec3<T>& iso, T z) {
        mqi::vec3<T> beam(0, 0, z);
        beam.x =
          iso.x * (treatment_machine_ion<T>::SAD_[0] - beam.z) / treatment_machine_ion<T>::SAD_[0];
        beam.y =
          iso.y * (treatment_machine_ion<T>::SAD_[1] - beam.z) / treatment_machine_ion<T>::SAD_[1];

        return beam;
    }
};

}   // namespace mqi

#endif
