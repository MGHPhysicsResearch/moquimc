#ifndef MQI_TREATMENT_SESSION_HPP
#define MQI_TREATMENT_SESSION_HPP

/// \file
///
/// Treatment session

#include <algorithm>
#include <type_traits>

#include "gdcmReader.h"

#include <moqui/base/mqi_beam_module_ion.hpp>
#include <moqui/base/mqi_ct.hpp>
#include <moqui/base/mqi_dataset.hpp>
#include <moqui/base/mqi_treatment_machine.hpp>
#include <moqui/base/mqi_treatment_machine_pbs.hpp>   //generic treatment machine
#include <moqui/base/mqi_utils.hpp>
//Add custom treatment machines

namespace mqi
{

/// \class treatment_session
/// \tparam T type of phase-space variables
/// Reads RT-Ion file, creates treatment_machine,
/// and returns machine objects, geometry, source, and coordinate system.
/// treatment_session is an entry point to RT-Ion interface to a MC engine.
/// \note we are considering to include patient, and dosegrid here.
template<typename T>
class treatment_session
{
protected:
    ///< Sequence tag dictionary for modality specific
    const std::map<const std::string, const gdcm::Tag>* seq_tags_;

    ///< RT Modality type, e.g., RTPLAN, RTRECORD, IONPLAN, IONRECORD
    mqi::modality_type mtype_;

    ///< machine name, e.g., institution_name:machine_name
    std::string machine_name_;

    mqi::treatment_machine<T>* tx_machine_ = nullptr;

    ///< top level DICOM dataset, either RTIP or RTIBTR
    mqi::dataset* mqi_ds_ = nullptr;

public:
    mqi::patient_material_t<T>* material_;

    /// Constructs treatment machine based on DICOM or specific file name.
    /// It reads in RT file recursively and construct a dataset tree
    /// Depending on RTIP or RTIBTR, it copies a propriate DICOM tag dictionaries
    /// (seqtags_per_modality).
    ///
    /// \param file_for_tx_machine : RTPLAN, IONPLAN, RTRECORD, IONRECORD
    /// Currently IONPLAN and IONRECORD are supported only.
    treatment_session(std::string file_for_tx_machine,   //for beamline and source,
                      std::string m_name      = "",
                      std::string calibration = "default") {

        gdcm::Reader reader;

        reader.SetFileName(file_for_tx_machine.c_str());

        const bool is_file_valid = reader.Read();
        if (!is_file_valid)
            throw std::runtime_error("Invalid DICOM file is given to treatment_session.");

        gdcm::MediaStorage ms;
        ms.SetFromFile(reader.GetFile());

        switch (ms) {
        case gdcm::MediaStorage::RTPlanStorage:
            mtype_ = RTPLAN;
            break;
        case gdcm::MediaStorage::RTIonPlanStorage:
            mtype_ = IONPLAN;
            break;
        case gdcm::MediaStorage::RTIonBeamsTreatmentRecordStorage:
            mtype_ = IONRECORD;
            break;
        default:
            throw std::runtime_error("treatment_session does not supports given RTMODALITY");
        }

        mqi_ds_   = new mqi::dataset(reader.GetFile().GetDataSet(), true);
        seq_tags_ = &mqi::seqtags_per_modality.at(mtype_);

        ///< machine name is set
        if (!m_name.compare("")) {
            ///< we assume machine name is in 1-st beam sequence
            ///< This part doesn't work for RTIBTR
            const mqi::dataset* beam_ds;
            if (mtype_ == IONPLAN) {
                beam_ds = (*mqi_ds_)(seq_tags_->at("beam"))[0];
            } else if (mtype_ == IONRECORD) {
                beam_ds = (*mqi_ds_)(seq_tags_->at("machine"))[0];
            } else {
                throw std::runtime_error(
                  "treatment_session can't find machine name and institute.");
            }

            std::vector<std::string> in;
            beam_ds->get_values("InstitutionName", in);
            std::vector<std::string> bn;
            beam_ds->get_values("TreatmentMachineName", bn);
            assert(bn.size() > 0);

            machine_name_ = (in.size() == 0) ? "" : mqi::trim_copy(in[0]);
            machine_name_ += ":" + mqi::trim_copy(bn[0]);

        } else {
            machine_name_ = m_name;
        }

        if (!this->create_machine(machine_name_, calibration)) {
            std::runtime_error("No MC machine is registered for " + machine_name_);
        }
    }

    /// Creates mqi::machine and return true for sucessful creation or false.
    /// \param machine_name for machine name
    /// \param mc_code for mc engine, e.g., code:version
    /// \note it takes itype_ member variables. caution, itype_ shouldn't be changed after creation.
    /// \note we have a branch for machines based on "string" comparison.
    ///  Looking for better way to determine during 'ideally' pre-processing.
    /// type_traits allows to branch the logic flow based on the type of variables.
    bool
    create_machine(std::string machine_name, std::string calibration) {
        if (tx_machine_) throw std::runtime_error("Preexisting machine.");

        std::cout << "machine_name: " << machine_name << ", calibration: " << calibration
                  << std::endl;

        const size_t deli = machine_name.find(":");

        std::string site = machine_name.substr(0, deli);
        std::transform(site.begin(), site.end(), site.begin(), ::tolower);
        std::transform(calibration.begin(), calibration.end(), calibration.begin(), ::tolower);

        std::string model = machine_name.substr(deli + 1, machine_name.size());

        if (site.compare("pbs"))
            std::transform(model.begin(), model.end(), model.begin(), ::tolower);

        std::cout << site << " : " << model << "\n";

        if (!site.compare("pbs")) {
            //Generic PBS beam model
            //expecting file
            std::cout << "Creating a generic PBS machine from : " << model << "\n";
            tx_machine_ = new mqi::pbs<T>(model);
            if (!calibration.compare("default")) {
                tx_machine_ = new mqi::pbs<T>(model);
            }else {
                throw std::runtime_error("Unknown calibration method. annony");
            }
            material_   = new mqi::pbs_material_t<T>();
            return true;
        } else {
            throw std::runtime_error("Valid machine is not available.");
        }

        return true;
    }

    /// Default destructor
    ~treatment_session() {
        delete tx_machine_;
        delete mqi_ds_;
    }

    /// Returns a list of beam names present in the plan file.
    std::vector<std::string>
    get_beam_names() {
        auto                     beam_sequence = (*mqi_ds_)(seq_tags_->at("beam"));
        std::vector<std::string> beam_names;
        for (const auto& beam : beam_sequence) {
            std::vector<std::string> tmp;
            beam->get_values("BeamName", tmp);
            beam_names.push_back(mqi::trim_copy(tmp[0]));
        }
        return beam_names;
    }

    /// Returns the number of beams present in the plan file.
    int
    get_num_beams() {
        auto beam_sequence = (*mqi_ds_)(seq_tags_->at("beam"));
        /// TODO: we need to count 'treatment' beam only. drop or not to use 'setup' beam.
        int count = 0;
        for (const auto& beam : beam_sequence) {
            count += 1;
        }
        return count;
    }

    int
    get_fractions() {
        auto             fraction_sequence = (*mqi_ds_)(gdcm::Tag(0x300a, 0x0070));
        std::vector<int> n_fractions;
        for (auto fraction : fraction_sequence) {
            fraction->get_values("NumberOfFractionsPlanned", n_fractions);
        }
        return n_fractions[0];
    }
    std::string
    get_beam_name(int bnb) {
        std::string beam_name;
        auto        bseq = (*mqi_ds_)(seq_tags_->at("beam"));
        for (auto i : bseq) {

            std::vector<int>         bn;
            std::vector<std::string> bname;
            i->get_values("BeamNumber", bn);   //works only for RTIP
            assert(bn.size() > 0);
            if (bnb == bn[0]) {
                i->get_values("BeamName", bname);
                beam_name = mqi::trim_copy(bname[0]);
                return beam_name;
            }
        }
        throw std::runtime_error("Invalid beam number.");
    }
    /// Search and return a beam (DICOM dataset) in BeamSequence for given beam name
    /// \param bnm for beam name
    /// \return mqi::dataset* constant pointer.
    const mqi::dataset*
    get_beam_dataset(std::string bnm) {
        auto bseq = (*mqi_ds_)(seq_tags_->at("beam"));
        for (auto i : bseq) {
            std::vector<std::string> bn;
            i->get_values("BeamName", bn);   //works for RTIP & RTIBTR
            if (bn.size() == 0) continue;
            if (bnm == mqi::trim_copy(bn[0])) {
                return i;
            }
        }
        throw std::runtime_error("Invalid beam name.");
    }

    /// Search and return a beam (DICOM dataset) in BeamSequence for given beam number
    /// \param bnm for beam number
    /// \return mqi::dataset* constant pointer.
    const mqi::dataset*
    get_beam_dataset(int bnb) {
        auto bseq = (*mqi_ds_)(seq_tags_->at("beam"));
        for (auto i : bseq) {
            //i->dump();
            std::vector<int> bn;
            i->get_values("BeamNumber", bn);   //works only for RTIP
            assert(bn.size() > 0);
            if (bnb == bn[0]) {
                return i;
            }
        }
        throw std::runtime_error("Invalid beam number.");
    }

    /// Get beamline object for given beam id, e.g, beam name or beam number
    /// \param beam_id for beam number or beam name
    /// \return beamline object.
    template<typename S>
    mqi::beamline<T>
    get_beamline(S beam_id) {
        return tx_machine_->create_beamline(this->get_beam_dataset(beam_id), mtype_);
    }

    /// Gets beam source object for given beam id, e.g, beam name or beam number
    /// \param beam_id for beam number or beam name
    /// \param coord   for coordinate transformation
    /// \param sid     for source to isocenter distance in mm
    /// \param scale   for calculating number of histories to be simulated from beamlet weight
    /// \return beamsource object.
    template<typename S>
    mqi::beamsource<T>
    get_beamsource(S beam_id, const mqi::coordinate_transform<T> coord, float scale, T sid) {
        return tx_machine_->create_beamsource(
          this->get_beam_dataset(beam_id), mtype_, coord, scale, sid);
    }

    /// Gets beam source object for given beam id, e.g, beam name or beam number
    /// \param beam_id for beam number or beam name
    /// \param coord   for coordinate transformation
    /// \param sid     for source to isocenter distance in mm
    /// \param scale   for calculating number of histories to be simulated from beamlet weight
    /// \return beamsource object.
    mqi::beamsource<T>
    get_beamsource(mqi::dataset*                      beam,
                   const mqi::coordinate_transform<T> coord,
                   float                              scale,
                   T                                  sid) {
        return tx_machine_->create_beamsource(beam, mtype_, coord, scale, sid);
    }

    /// Gets time line object for given beam id, e.g., beam name or number
    /// \param beam_id
    template<typename S>
    std::map<T, int32_t>
    get_timeline(S beam_id) {
        return tx_machine_->create_timeline(this->get_beam_dataset(beam_id), mtype_);
    }

    /// Gets beam coordinate object for given beam id, e.g, beam name or beam number
    /// \param beam_id for beam number or beam name
    /// \return beam coordinate
    template<typename S>
    mqi::coordinate_transform<T>
    get_coordinate(S beam_id) {
        return tx_machine_->create_coordinate_transform(this->get_beam_dataset(beam_id), mtype_);
    }

    /// Summarize plan
    /// \param bnb for beam number
    void
    summary(void) {
        //plan_ds
        mqi_ds_->dump();
    }

    /// Return ion type
    /// \return mqi::m_type_
    mqi::modality_type
    get_modality_type(void) {
        return mtype_;
    }

    const gdcm::DataElement&
    get_dataelement(gdcm::Tag tag) {
        return mqi_ds_[0][tag];
    }
};

}   // namespace mqi

#endif
