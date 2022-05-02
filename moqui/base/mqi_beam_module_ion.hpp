#ifndef MQI_BEAM_MODULE_ION_H
#define MQI_BEAM_MODULE_ION_H

/// \file
///
/// Interprets DICOM-RT Ion beam module
/// \see http://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.8.25.html for RTI
/// \see http://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.8.26.html for RTIBTR

#include <moqui/base/mqi_beam_module.hpp>

namespace mqi
{

/// Scan type from a tag (300A,0308)
typedef enum
{
    NONE           = 0,
    UNIFORM        = 1,
    MODULATED      = 2,
    MODULATED_SPEC = 3
} SCAN_MODE;

/// \class beam_module_ion
///  A class for the RTION beams (plan & treatment record)

class beam_module_ion : public beam_module
{
public:
    /// User-defined type for a spot
    typedef struct {
        float e;          ///< The energy in MeV
        float x;          ///< The x-position in mm
        float y;          ///< The y-position in mm
        float fwhm_x;     ///< The full-width-half maximum for x
        float fwhm_y;     ///< The full-width-hafl maximum for y
        float meterset;   ///< The meterset weight, e.g.,MU or NP.
    } spot;

protected:
    /// Number of spots as a function of layer-id
    std::vector<int> nb_spots_per_layer_;

    /// All spots in delivery order.
    std::vector<spot> sequence_;

    /// Name of beam model
    std::string tune_id_;

public:
    /// Constructs beam module for RT-Ion
    /// \param d DICOM dataset of either C.8.8.25-1 (plan) or C.8.8.26-1 (record)
    /// \param m a modality type, e.g., ION Plan or ION Record
    beam_module_ion(const mqi::dataset* d, mqi::modality_type m) : beam_module(d, m) {
        /// Initializes containers
        std::vector<int>         nb_pts(1);
        std::vector<float>       energy(1);
        std::vector<float>       fwhm_xy(2);
        std::vector<std::string> tune_id(1);
        std::vector<float>       xy;
        std::vector<float>       weight;

        int layer_nb = 0;

        std::string str_weight;

        /// Checks modality type
        switch (modality_) {
        case mqi::modality_type::IONPLAN:
            str_weight = "ScanSpotMetersetWeights";
            break;
        case mqi::modality_type::IONRECORD:
            str_weight = "ScanSpotMetersetsDelivered";
            break;
        default:
            throw std::runtime_error("Wrong ION type");
        }

        /// Fill the containers from the DICOM dataset
        /// As each layer consists of pair of ion-controls and even-layers have all zero weights.
        /// So we drop even layer.
        auto seq_tags = &mqi::seqtags_per_modality.at(m);
        auto ictrl    = (*ds_)(seq_tags->at("ctrl"));
        for (auto b : ictrl) {

            if ((layer_nb++) % 2) continue;

            b->get_values("ScanSpotTuneID", tune_id);
            b->get_values("NominalBeamEnergy", energy);
            b->get_values("NumberOfScanSpotPositions", nb_pts);
            b->get_values("ScanningSpotSize", fwhm_xy);
            b->get_values("ScanSpotPositionMap", xy);
            b->get_values(str_weight.c_str(), weight);

            for (int j = 0; j < nb_pts[0]; ++j) {
                tune_id_ = tune_id[0];
                sequence_.push_back(
                  { energy[0], xy[j * 2], xy[j * 2 + 1], fwhm_xy[0], fwhm_xy[1], weight[j] });
            }   //per spot
            nb_spots_per_layer_.push_back(nb_pts[0]);
        }   //per layer
    }

    /// Destructor
    ~beam_module_ion() {
        ;
    }

    /// Returns the vector pointer for number of spots as a function of layer-id
    const std::vector<int>*
    get_nb_spots_per_layer(void) const {
        return &nb_spots_per_layer_;
    }

    /// Returns a pointer of spot-sequence vector
    const std::vector<spot>*
    get_sequence(void) const {
        return &sequence_;
    }

    /// Returns tune-id
    const std::string
    get_tune_id(void) const {
        return tune_id_;
    }

    /// Prints out spot-sequence
    void
    dump() const {
        std::cout << "dump:spotmap, size:" << sequence_.size() << std::endl;
        for (auto i : sequence_) {
            std::cout << "spot (E,X,Y,Sx,Sy,W): " << i.e << ", " << i.x << ", " << i.y << ", "
                      << i.fwhm_x << ", " << i.fwhm_y << ", " << i.meterset << std::endl;
        }
    }
};

}   // namespace mqi

#endif
