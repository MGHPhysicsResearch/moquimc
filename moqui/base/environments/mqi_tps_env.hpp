#ifndef MQI_TPS_RAY_ENV_HPP
#define MQI_TPS_RAY_ENV_HPP

#include <moqui/base/environments/mqi_xenvironment.hpp>
#include <moqui/base/scorers/mqi_scorer_energy_deposit.hpp>

#include "gdcmAttribute.h"
#include "gdcmDataElement.h"
#include "gdcmDataSet.h"
#include "gdcmDict.h"
#include "gdcmDicts.h"
#include "gdcmGlobal.h"
#include "gdcmIPPSorter.h"
#include "gdcmImage.h"
#include "gdcmReader.h"
#include "gdcmScanner.h"
#include "gdcmSorter.h"
#include "gdcmStringFilter.h"
#include "gdcmTag.h"
#include "gdcmTesting.h"
#include <cassert>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>
#include <moqui/base/materials/mqi_patient_materials.hpp>
#include <moqui/base/mqi_aperture.hpp>
#include <moqui/base/mqi_aperture3d.hpp>
#include <moqui/base/mqi_distributions.hpp>
#include <moqui/base/mqi_file_handler.hpp>
#include <moqui/base/mqi_io.hpp>
#include <moqui/base/mqi_math.hpp>
#include <moqui/base/mqi_rangeshifter.hpp>
#include <moqui/base/mqi_roi.hpp>
#include <moqui/base/mqi_threads.hpp>
#include <moqui/base/mqi_treatment_session.hpp>
#include <moqui/base/scorers/mqi_scorer_energy_deposit.hpp>
#include <valarray>

namespace mqi
{
struct dicom_t {
    mqi::vec3<ijk_t>               dim_;       //number of voxels
    mqi::vec3<ijk_t>               org_dim_;   //number of voxels
    float                          dx = -1;
    float                          dy = -1;
    float*                         org_dz;
    float*                         dz;
    uint16_t                       num_vol  = 0;
    uint16_t                       nfiles   = 0;
    uint16_t                       n_plan   = 0;
    uint16_t                       n_dose   = 0;
    uint16_t                       n_struct = 0;
    float*                         xe       = nullptr;
    float*                         ye       = nullptr;
    float*                         ze       = nullptr;
    float*                         org_xe   = nullptr;
    float*                         org_ye   = nullptr;
    float*                         org_ze   = nullptr;
    gdcm::Directory::FilenamesType plan_list;
    gdcm::Directory::FilenamesType dose_list;
    gdcm::Directory::FilenamesType struct_list;
    gdcm::Directory::FilenamesType ct_list;
    std::string                    plan_name   = "";
    std::string                    struct_name = "";
    std::string                    dose_name   = "";
    mqi::ct<phsp_t>*               ct;
    mqi::vec3<float>               image_center;
    mqi::vec3<size_t>              dose_dim;
    mqi::vec3<float>               dose_pos0;
    float                          dose_dx;
    float                          dose_dy;
    float*                         dose_dz;
    mqi::vec3<uint16_t>            clip_shift_;
    uint8_t*                       body_contour;
    uint8_t*                       roi_contour;
};
template<typename R>
class tps_env : public x_environment<R>
{
public:
    std::string input_filename   = "";
    std::string delimeter        = " ";
    int         master_seed      = 0;
    std::string machine_name     = "";
    std::string calibration_name = "";
    bool        score_to_ct_grid = true;
    std::string output_path      = "";
    std::string output_format =
      "";   /// currently support mhd, other than mhd are considered as binary
    /// Scorer parameters
    std::vector<std::string>   scorer_string;
    std::vector<mqi::scorer_t> scorer_type;
    bool                       score_variance = false;
    std::string                source_type    = "FluenceMap";
    /// Simulation parameters
    mqi::sim_type_t            sim_type;
    std::vector<int>           beam_numbers;
    std::string                parent_dir = "";
    std::string                dicom_dir  = "";
    std::string                ct_name    = "";
    std::string                ct_path    = "";
    std::vector<std::string>   parameters_total;
    std::vector<std::string>   mask_filenames;
    bool                       scoring_mask      = false;
    bool                       overwrite_results = false;
    bool                       use_absolute_path;
    size_t                     max_histories_per_batch;
    bool                       memory_save_mode;
    bool                       save_scorer_map;
    std::string                scorer_map_prefix;
    dicom_t                    dcm_;
    int16_t*                   ct_data;
    mqi::treatment_session<R>* tx;
    uint16_t                   bnb                   = 0;
    float                      sid                   = 0.0;
    float                      particles_per_history = -1.0;
    std::string                beam_prefix;
    density_t*                 stopping_power;
    float                      max_let_in_water;
    int                        aperture_ind  = -1;
    mqi::aperture_type_t       aperture_type = mqi::VOLUME;
    std::vector<float>         scorer_voxel_size;
    bool                       ct_clipping;
    int                        verbosity;
    std::string                body_contour_name;
    bool                       read_structure;
    uint32_t                   scorer_size;
    uint32_t                   scorer_capacity;
    bool                       reshape_output       = false;
    bool                       sparse_output        = false;
    std::string                source_extension     = "";
    float                      rbe                  = 0;
    int                        n_fractions          = 0;
    int                        unit_weights         = 0;
    float                      rangeshifter_density = 0;
    //// Robustness setting
    float density_scale  = 0;
    float volume_shift_x = 0;
    float volume_shift_y = 0;
    float volume_shift_z = 0;
    //// Statistical stopping criteria
    bool                     record_statistics          = false;
    bool                     save_statistics            = false;
    bool                     set_stat_roi_from_rtstruct = false;
    float                    stat_criteria              = 0;
    std::string              stat_roi                   = "";
    std::vector<std::string> stat_roi_mask_filenames;
    int                      stat_roi_size  = 0;
    float                    stat_threshold = 0.0;
    //    std::default_random_engine beam_rng;

public:
    CUDA_HOST
    tps_env(const std::string input_name) : x_environment<R>() {
        struct stat info;
        std::string delimeter = " ";
        input_filename        = input_name;
        mqi::file_parser parser(input_filename, delimeter);
        /// Global parameters
        this->gpu_id = parser.get_int("GPUID", 0);
#if defined(__CUDACC__)
        cudaSetDevice(this->gpu_id);
#endif
        master_seed = parser.get_int("RandomSeed", -1);
        if (master_seed == -1) {
            master_seed = std::time(nullptr);
        }
        printf("master seed %d\n", master_seed);
        use_absolute_path = parser.get_bool("UseAbsolutePath", false);

        this->num_total_threads = parser.get_int("TotalThreads", -1);
        beam_prefix             = parser.get_string("BeamPrefix", "beam");
        max_histories_per_batch = parser.get_int("MaxHistoriesPerBatch", 0);

        /// Data directories
        if (use_absolute_path) {
            this->parent_dir = parser.get_string("ParentDir", "");
            this->dicom_dir  = parser.get_string("DicomDir", "");
            this->ct_name    = parser.get_string("CTVolumeName", "");
            this->ct_path    = this->ct_name;
        } else {
            this->parent_dir = parser.get_string("ParentDir", "");
            if (parent_dir.empty()) {
                throw std::runtime_error("ParentDir is not provided");
            }
            this->dicom_dir = this->parent_dir + "/" + parser.get_string("DicomDir", "");
            this->ct_name   = parser.get_string("CTVolumeName", "");
            this->ct_path   = this->parent_dir + "/" + this->ct_name;
        }
        if (parent_dir.empty()) {
            throw std::runtime_error("ParentDir is not provided");
        }
        if (dicom_dir.empty()) {
            throw std::runtime_error("ParentDir is not provided");
        }

        /// Source parameters
        this->source_type = parser.get_string("SourceType", "FluenceMap");
        this->sim_type = parser.string_to_sim_type(parser.get_string("SimulationType", "perBeam"));
        this->particles_per_history = parser.get_float("ParticlesPerHistory", -1.0);
        this->rbe                   = parser.get_float("RBE", 1.1);
        this->n_fractions           = parser.get_int("NumberOfFraction", -1);
        this->rangeshifter_density  = parser.get_float("RangeshifterDensity", 1.19);
        this->unit_weights          = parser.get_int("UnitWeights", -1);
        this->machine_name          = parser.get_string("Machine", "");
        this->calibration_name      = parser.get_string("Calibration", "default");
        /// Scorer parameters
        this->scorer_string = parser.get_string_vector("Scorer", ",");
        this->scorer_type   = parser.strings_to_scorer_types(this->scorer_string);

        //// Set simulation type to per spot for dose dij matrix scoring
        for (int ii = 0; ii < this->scorer_type.size(); ii++) {
            if (this->scorer_type[ii] == mqi::DOSE_Dij) {
                this->sim_type = mqi::PER_SPOT;
                if (this->scorer_type.size() > 1) {
                    throw std::runtime_error("Dij cannot be scored with the other quantities");
                }
            }
        }

        //        score_variance          = !parser.get_bool("SupressStd", true);
        score_to_ct_grid        = parser.get_bool("ScoreToCTGrid", true);
        scoring_mask            = parser.get_bool("ScoringMask", false);
        ct_clipping             = false;   //parser.get_bool("CTClipping", false);
        this->body_contour_name = parser.get_string("BodyContourName", "External");
        this->read_structure    = parser.get_bool("ReadStructure", false);
        if (scoring_mask) {
            save_scorer_map = parser.get_bool("SaveMap", true);
            mask_filenames  = parser.get_string_vector("Mask", ",");
            if (mask_filenames.size() == 0) {
                throw std::runtime_error("Mask filename is missing");
            }
        } else {
            save_scorer_map = false;
            mask_filenames  = {};
        }
        if (save_scorer_map) {
            scorer_map_prefix = parser.get_string("ScorerMapName", "scorer_map");
        } else {
            scorer_map_prefix = "";
        }
        score_to_ct_grid = true;
        if (!score_to_ct_grid) {
            /// TODO: dose grid scoring
            scorer_voxel_size = parser.get_float_vector("ScorerVoxelSize", ",");
            if (scorer_voxel_size.size() == 0) {
                scorer_voxel_size = { 0, 0, 0 };
            }
        }

        //// Robust options
        this->density_scale  = parser.get_float("DensityScaling", 1);
        this->volume_shift_x = parser.get_float("XShift", 0);
        this->volume_shift_y = parser.get_float("YShift", 0);
        this->volume_shift_z = parser.get_float("ZShift", 0);

        /// Output parameters
        output_path   = parser.get_string("OutputDir", "");
        output_format = parser.get_string("OutputFormat", "raw");
        if (strcasecmp(output_format.c_str(), "npz") == 0) {
            this->reshape_output = false;
            this->sparse_output  = true;
        } else {
            this->reshape_output = true;
            this->sparse_output  = false;
        }
        if (output_path.empty()) {
            throw std::runtime_error("Output directory is not provided.");
        }
        overwrite_results = parser.get_bool("OverwriteResults", false);
        if (stat(output_path.c_str(), &info) != 0) {
            mkdir(output_path.c_str(), 0755);
        } else if (!overwrite_results) {
            throw std::runtime_error("Output directory exists.");
        }

        /// Uncertainty stopping criteria
        this->record_statistics = parser.get_bool("StoppingStatistics", false);
        this->save_statistics   = parser.get_bool("SaveStoppingStatistics", false);

        if (this->record_statistics) {
            this->stat_criteria = parser.get_float("StoppingCriteria", -1.0);
            if (this->stat_criteria < 0) {
                throw std::runtime_error("Statistical criteria must be positive float");
            }
        }

        this->stat_roi_mask_filenames    = parser.get_string_vector("StatROIMaskFilename", ",");
        this->set_stat_roi_from_rtstruct = parser.get_bool("StatROIStructFromRT", false);
        if (this->set_stat_roi_from_rtstruct) {
            this->stat_roi = parser.get_string("StatROI", "");
            if (stat_roi.empty()) {
                throw std::runtime_error("Statistical ROI is not provided.");
            }
        }
        std::cout << parent_dir << std::endl;

        this->stat_threshold = parser.get_float("StatThreshold", 0.0);
        if (this->record_statistics) {
            if (!this->set_stat_roi_from_rtstruct && this->stat_roi_mask_filenames.size() == 0) {
                if (this->stat_threshold <= 0) {
                    printf(
                      "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!ERROR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
                    printf(
                      "If no contour or mask is selected, the statical threshold cannot be zero\n");
                    printf(
                      "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!ERROR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
                }
                assert(this->stat_threshold > 0);
            } else if (this->stat_threshold <= 0) {
                printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!WARNING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
                printf("If statistical threshold is zero, the uncertainty might be biased\n");
                printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!WARNING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
            }
        }

        ///Initialize data
        this->dcm_ = this->read_dcm_dir();
        printf("%s\n", dcm_.plan_name.c_str());
        if (!this->machine_name.empty()) {
            printf("machine force selection %s\n", this->machine_name.c_str());
            tx = new mqi::treatment_session<R>(
              dcm_.plan_name, this->machine_name.c_str(), this->calibration_name);
        } else {
            tx = new mqi::treatment_session<R>(dcm_.plan_name, "", this->calibration_name);
        }
        if (this->n_fractions <= 0) {
            this->n_fractions = tx->get_fractions();
        }
        printf("n fraction %d rbe %f\n", this->n_fractions, this->rbe);
        if (sim_type == mqi::PER_BEAM) {
            beam_numbers = parser.get_int_vector("BeamNumbers", ",");
            printf("Number of beam selected %d\n", beam_numbers.size());
            printf("The first selected beam number %d\n", beam_numbers[0]);
            if (beam_numbers.size() == 0) {
                for (int k = 1; k < this->tx->get_num_beams() + 1; k++) {
                    //                    printf("%s\n", this->tx->get_beam_name(k).c_str());
                    if (this->tx->get_beam_name(k) == "Setup") continue;
                    beam_numbers.push_back(k);
                }
            } else if (beam_numbers.size() == 1 && beam_numbers[0] == 0) {
                beam_numbers.clear();
                for (int k = 1; k < this->tx->get_num_beams() + 1; k++) {
                    if (this->tx->get_beam_name(k) == "Setup") continue;
                    beam_numbers.push_back(k);
                }
            }
        } else if (sim_type == mqi::PER_SPOT) {
            ct_clipping  = false;
            beam_numbers = parser.get_int_vector("BeamNumbers", ",");
            if (beam_numbers.size() == 0) {
                for (int k = 1; k < this->tx->get_num_beams() + 1; k++) {
                    if (this->tx->get_beam_name(k) == "Setup") continue;
                    beam_numbers.push_back(k);
                }
            } else if (beam_numbers.size() == 1 && beam_numbers[0] == 0) {
                beam_numbers.clear();
                for (int k = 1; k < this->tx->get_num_beams() + 1; k++) {
                    if (this->tx->get_beam_name(k) == "Setup") continue;
                    beam_numbers.push_back(k);
                }
            }
        } else if (sim_type == mqi::PER_PATIENT) {
            beam_numbers.clear();
            for (int k = 1; k < this->tx->get_num_beams() + 1; k++) {
                beam_numbers.push_back(k);
            }
        }
    }

    CUDA_HOST
    ~tps_env() {
        ;
    }

    CUDA_HOST
    virtual void
    print_parameters() {
        printf("================================\n");
        printf("Global parameters\n");
        printf("================================\n");
        printf("Input parameter file: %s\n", input_filename.c_str());
        printf("GPU_ID %d\n", this->gpu_id);
        printf("Random seed %d\n", master_seed);
        printf("The number of total threads %d\n", this->num_total_threads);
        printf("Maximum histories per batch %lu\n", max_histories_per_batch);
        printf("================================\n");
        printf("Setup parameters\n");
        printf("================================\n");
        printf("Patient directory %s\n", parent_dir.c_str());
        printf("DICOM directory %s\n", dicom_dir.c_str());
        //        printf("CT directory %s\n", ct_name.c_str());
        printf("Scorer type %d\n", this->scorer_type);
        printf("Supress variance %d\n", !score_variance);
        printf("Particles per histories %.1f\n", particles_per_history);
        printf("Source type %s\n", source_type.c_str());
        printf("Simulation type %d\n", sim_type);
        if (sim_type == mqi::PER_BEAM) {
            printf("Beam numbers ");
            for (int i = 0; i < beam_numbers.size(); i++) {
                printf("%d ", beam_numbers[i]);
            }
            printf("\n");
        }
        printf("Machine name %s\n", machine_name.c_str());
        printf("Score CT grid %d\n", score_to_ct_grid);
        printf("Scoring mask %d\n", scoring_mask);
        printf("Save scorer map %d\n", save_scorer_map);
        if (save_scorer_map) {
            printf("Scorer map save prefix %s\n", scorer_map_prefix.c_str());
        }
        printf("Using absolute path %d\n", use_absolute_path);
        printf("Beam prefix %s\n", beam_prefix.c_str());
        printf("================================\n");
        printf("Output parameters\n");
        printf("================================\n");
        printf("Output path %s\n", output_path.c_str());
        printf("Output format %s\n", output_format.c_str());
        printf("Overwrite output %d\n", overwrite_results);

        if (scoring_mask) {
            printf("Mask filenames\n");
            for (int i = 0; i < mask_filenames.size(); i++) {
                printf("%s\n", mask_filenames[i].c_str());
            }
        }
    }

    CUDA_HOST
    virtual struct dicom_t
    read_dcm_dir() {
        dicom_t         dcm;
        gdcm::Directory d;
        //// Need to check the directory is valid and stop process if not
        printf("dicom dir %s\n", dicom_dir.c_str());
        d.Load(dicom_dir.c_str());
        const gdcm::Directory::FilenamesType& l1 = d.GetFilenames();
        dcm.nfiles                               = l1.size();
        gdcm::Scanner   s0;
        const gdcm::Tag Modality(0x0008, 0x0060);   // Modality
        s0.AddTag(Modality);
        bool b = s0.Scan(d.GetFilenames());
        if (!b) {
            std::cerr << "Scanner failed" << std::endl;
            throw std::runtime_error("Reading DICOM failed.");
            //            return;
        }
        // Only get the DICOM files:
        dcm.plan_list   = s0.GetAllFilenamesFromTagToValue(Modality, "RTPLAN");
        dcm.struct_list = s0.GetAllFilenamesFromTagToValue(Modality, "RTSTRUCT");

        printf("ct name: %s\n", this->ct_name.c_str());
        float xe0, ye0, ze0;
        if (this->ct_name.empty()) {
            dcm.ct = new mqi::ct<R>(dicom_dir, false);
            dcm.ct->load_data();
            dcm.dim_      = dcm.ct->get_nxyz();
            dcm.org_dim_  = dcm.ct->get_nxyz();
            dcm.dx        = dcm.ct->get_dx();
            dcm.dy        = dcm.ct->get_dy();
            dcm.org_dz    = dcm.ct->get_dz();
            xe0           = dcm.ct->get_x()[0] - dcm.dx / 2.0;
            ye0           = dcm.ct->get_y()[0] - dcm.dy / 2.0;
            ze0           = dcm.ct->get_z()[0] - dcm.org_dz[0] / 2.0;
            this->ct_data = new int16_t[dcm.org_dim_.x * dcm.org_dim_.y * dcm.org_dim_.z];
            std::copy(begin(dcm.ct->get_data()), end(dcm.ct->get_data()), this->ct_data);
        } else {
            mqi::vec3<float> ct_origin = read_ct_image(dcm);
            xe0                        = ct_origin.x - dcm.dx / 2.0;
            ye0                        = ct_origin.y - dcm.dy / 2.0;
            ze0                        = ct_origin.z - dcm.org_dz[0] / 2.0;
            printf("dcm.dim nx %d ny %d nz %d\n", dcm.org_dim_.x, dcm.org_dim_.y, dcm.org_dim_.z);
        }
        xe0 += this->volume_shift_x;
        ye0 += this->volume_shift_y;
        ze0 += this->volume_shift_z;

        dcm.n_plan   = dcm.plan_list.size();
        dcm.n_struct = dcm.struct_list.size();

        assert(dcm.n_plan < dcm.nfiles);
        if (dcm.n_plan > 0) {
            if (dcm.n_plan >= 1) {
                printf("There are multiple RTPLANs in the path. Selecting the first one\n");
            }
            dcm.plan_name = dcm.plan_list[0];
        } else {
            dcm.plan_name = "";
            throw std::runtime_error("There is no RTPLAN in the path. Exiting.");
        }

        if (dcm.n_struct > 0) {
            if (dcm.n_struct > 1) {
                printf("There are multiple RTSTRUCTs in the path. Selecting the first one\n");
            }
            dcm.struct_name = dcm.struct_list[0];
        } else {
            dcm.struct_name = "";
        }
        printf("Loading RT Ion plan from %s\n", dcm.plan_name.c_str());
        if (dcm.n_plan > dcm.nfiles || dcm.n_dose > dcm.nfiles || dcm.n_struct > dcm.nfiles ||
            dcm.dim_.z > dcm.nfiles) {
            throw std::runtime_error("something wrong in reading the directory.");
        }
        // array of the dcm.ct contains center of voxels
        dcm.org_xe = new float[dcm.dim_.x + 1];
        dcm.org_ye = new float[dcm.dim_.y + 1];
        dcm.org_ze = new float[dcm.dim_.z + 1];

        // Change the voxel center position to edge positions
        for (int i = 0; i < dcm.dim_.x + 1; i++)
            dcm.org_xe[i] = xe0 + i * dcm.dx;
        for (int i = 0; i < dcm.dim_.y + 1; i++) {
            dcm.org_ye[i] = ye0 + i * dcm.dy;
        }

        for (int i = 0; i < dcm.dim_.z + 1; i++) {
            if (i == 0)
                dcm.org_ze[i] = ze0;
            else
                dcm.org_ze[i] =
                  dcm.org_ze[i - 1] +
                  dcm.org_dz[i - 1];   // need to be modified to deal with varying slice thickness
        }
        dcm.image_center.x =
          (dcm.org_xe[0] + dcm.dx / 2.0 + dcm.org_xe[dcm.dim_.x] - dcm.dx / 2.0) / 2.0;
        dcm.image_center.y =
          (dcm.org_ye[0] + dcm.dy / 2.0 + dcm.org_ye[dcm.dim_.y] - dcm.dy / 2.0) / 2.0;
        dcm.image_center.z = (dcm.org_ze[0] + dcm.org_dz[0] / 2.0 + dcm.org_ze[dcm.dim_.z] -
                              dcm.org_dz[dcm.dim_.z - 1] / 2.0) /
                             2.0;
        std::chrono::time_point<std::chrono::high_resolution_clock> start, stop;
        std::chrono::duration<double, std::milli>                   duration;

        //// TODO: ct clipping check
        this->ct_clipping = false;
        if (this->ct_clipping) {
            //// TODO: ct clipping check
            //            dcm = this->clip_ct(dcm);
        } else {
            dcm.xe        = dcm.org_xe;
            dcm.ye        = dcm.org_ye;
            dcm.ze        = dcm.org_ze;
            dcm.dz        = dcm.org_dz;
            this->ct_data = new int16_t[dcm.org_dim_.x * dcm.org_dim_.y * dcm.org_dim_.z];
            std::copy(begin(dcm.ct->get_data()), end(dcm.ct->get_data()), this->ct_data);
        }
        if (dcm.n_struct >= 1 && this->read_structure) {
            printf("Loading RTSTRUCT from %s\n", dcm.struct_name.c_str());
            gdcm::Reader struct_reader;
            struct_reader.SetFileName(dcm.struct_name.c_str());
            const bool is_valid_rs = struct_reader.Read();
            if (!is_valid_rs) throw std::runtime_error("Invalid RTSTRUCT file.");
            mqi::dataset* struct_ds_ = new mqi::dataset(struct_reader.GetFile().GetDataSet(), true);
            auto          roi_seq    = (*struct_ds_)(gdcm::Tag(0x3006, 0x0020));
            printf("roi seq size %lu\n", roi_seq.size());
            std::vector<std::string> roi_name;
            std::vector<int>         body_ind;

            for (int roi_ind = 0; roi_ind < roi_seq.size(); roi_ind++) {
                roi_seq[roi_ind]->get_values("ROIName", roi_name);
                if (strcasecmp(roi_name[0].c_str(), this->body_contour_name.c_str()) == 0) {
                    printf("%s\n", roi_name[0].c_str());
                    roi_seq[roi_ind]->get_values("ROINumber", body_ind);
                    printf("external id %d\n", body_ind[0]);
                    break;
                }
            }
            auto     roi_contour_seq = (*struct_ds_)(gdcm::Tag(0x3006, 0x0039));
            uint8_t* body_contour = new uint8_t[dcm.org_dim_.x * dcm.org_dim_.y * dcm.org_dim_.z];
            std::vector<int>   refer_roi, contour_num;
            std::vector<float> contour_data;
            mqi::vec3<float>*  contour_points;
            int                points_ind;
            start = std::chrono::high_resolution_clock::now();
            for (int con_ind = 0; con_ind < roi_contour_seq.size(); con_ind++) {
                roi_contour_seq[con_ind]->get_values("ReferencedROINumber", refer_roi);
                if (refer_roi[0] == body_ind[0]) {
                    auto contour_seq = (*roi_contour_seq[con_ind])(gdcm::Tag(0x3006, 0x0040));
                    for (int contour_ind = 0; contour_ind < contour_seq.size(); contour_ind++) {
                        contour_seq[contour_ind]->get_values("NumberOfContourPoints", contour_num);
                        contour_seq[contour_ind]->get_values("ContourData", contour_data);
                        contour_points = new mqi::vec3<float>[contour_num[0]];
                        for (points_ind = 0; points_ind < contour_num[0]; points_ind++) {
                            contour_points[points_ind].x = contour_data[points_ind * 3];
                            contour_points[points_ind].y = contour_data[points_ind * 3 + 1];
                            contour_points[points_ind].z = contour_data[points_ind * 3 + 2];
                        }
                        fill_contour(body_contour,
                                     contour_points,
                                     contour_num[0],
                                     dcm.dim_,
                                     dcm.xe,
                                     dcm.ye,
                                     dcm.ze,
                                     dcm.dx,
                                     dcm.dy);
                        delete[] contour_points;
                    }
                    break;
                }
            }
            stop     = std::chrono::high_resolution_clock::now();
            duration = stop - start;
            printf("Contour conversion to volume: %f s\n", duration.count() * 0.0001);
            dcm.body_contour = body_contour;
        } else if (this->read_structure) {
            throw std::runtime_error("RT STRCUTURE does not exist");
        }
        return dcm;
    }

    CUDA_HOST
    virtual mqi::vec3<float>
    read_ct_image(dicom_t& dcm) {
        mqi::vec3<float> origin;
        std::string      line;
        std::ifstream    fid(this->ct_path, std::ios::ate);
        fid.seekg(0);
        std::string ct_delimeter = "=";
        size_t      pos, pos_current, pos_prev, total_size;
        std::string parameter, value;
        uint16_t    nx, ny, nz;
        float       dx, dy, dz1;
        float*      dz;
        while (std::getline(fid, line)) {
            line      = trim_copy(line);
            pos       = line.find(ct_delimeter);
            parameter = trim_copy(line.substr(0, pos));
            if (strcasecmp(parameter.c_str(), "ObjectType") == 0) {
            } else if (strcasecmp(parameter.c_str(), "NDims") == 0) {
            } else if (strcasecmp(parameter.c_str(), "BinaryData") == 0) {
            } else if (strcasecmp(parameter.c_str(), "BinaryDataByteOrderMSB") == 0) {
            } else if (strcasecmp(parameter.c_str(), "CompressedData") == 0) {
            } else if (strcasecmp(parameter.c_str(), "TransformMatrix") == 0) {
            } else if (strcasecmp(parameter.c_str(), "Offset") == 0) {
                value       = trim_copy(line.substr(pos + 1));
                pos_prev    = 0;
                pos_current = value.find(" ");
                origin.x    = std::atof(value.substr(pos_prev, pos_current).c_str());
                pos_prev    = pos_current + 1;
                pos_current = value.find(" ", pos_prev);
                origin.y    = std::atof(value.substr(pos_prev, pos_current).c_str());
                pos_prev    = pos_current + 1;
                pos_current = value.find(" ", pos_prev);
                origin.z    = std::atof(value.substr(pos_prev, pos_current).c_str());
            } else if (strcasecmp(parameter.c_str(), "CenterOfRotation") == 0) {
            } else if (strcasecmp(parameter.c_str(), "AnatomicalOrientation") == 0) {
            } else if (strcasecmp(parameter.c_str(), "ElementSpacing") == 0) {
                value       = trim_copy(line.substr(pos + 1));
                pos_prev    = 0;
                pos_current = value.find(" ");
                dx          = std::atof(value.substr(pos_prev, pos_current).c_str());
                pos_prev    = pos_current + 1;
                pos_current = value.find(" ", pos_prev);
                dy          = std::atof(value.substr(pos_prev, pos_current).c_str());
                pos_prev    = pos_current + 1;
                pos_current = value.find(" ", pos_prev);
                dz1         = std::atof(value.substr(pos_prev, pos_current).c_str());
            } else if (strcasecmp(parameter.c_str(), "DimSize") == 0) {
                value       = trim_copy(line.substr(pos + 1));
                pos_prev    = 0;
                pos_current = value.find(" ");
                nx          = std::atoi(value.substr(pos_prev, pos_current).c_str());
                pos_prev    = pos_current + 1;
                pos_current = value.find(" ", pos_prev);
                ny          = std::atoi(value.substr(pos_prev, pos_current).c_str());
                pos_prev    = pos_current + 1;
                pos_current = value.find(" ", pos_prev);
                nz          = std::atoi(value.substr(pos_prev, pos_current).c_str());
            } else if (strcasecmp(parameter.c_str(), "ElementType") == 0) {
            } else if (strcasecmp(parameter.c_str(), "ElementDataFile") == 0) {
                value = trim_copy(line.substr(pos + 1));
                if (strcasecmp(value.c_str(), "LOCAL") != 0) {
                    throw std::runtime_error("Mask files does not contain data.");
                }
                break;   // This break is necessary to read correct number of voxels
            }
        }
        dz = new float[nz];
        for (int kk = 0; kk < nz; kk++) {
            dz[kk] = dz1;
        }
        total_size = nx * ny * nz;
        //Assuming it's float
        float* ct_tmp = new float[total_size];
        fid.read((char*) (&ct_tmp[0]), total_size * sizeof(ct_tmp[0]));
        fid.close();
        this->ct_data = new int16_t[total_size];
        for (int kk = 0; kk < total_size; kk++) {
            this->ct_data[kk] = (int16_t) ct_tmp[kk];
        }
        delete[] ct_tmp;
        dcm.dim_     = mqi::vec3<mqi::ijk_t>(nx, ny, nz);
        dcm.org_dim_ = mqi::vec3<mqi::ijk_t>(nx, ny, nz);
        dcm.dx       = dx;
        dcm.dy       = dy;
        dcm.org_dz   = dz;
        return origin;
    }

    CUDA_HOST
    virtual void
    setup_materials() {
        this->n_materials = 1;
        this->materials   = new mqi::patient_material_t<R>;
        this->materials   = this->tx->material_;
    }

    CUDA_HOST
    virtual void
    setup_world() {
        this->world = new mqi::node_t<R>;
        ///< By default, let's set +- 30 cm as world volume
        //		const R hl   = 600.0;
        const R hl   = 800.0;
        R       x[2] = { this->dcm_.org_xe[0] - (float) 0.5 * hl,
                         this->dcm_.org_xe[this->dcm_.org_dim_.x] + (float) 0.5 * hl };
        R       y[2] = { this->dcm_.org_ye[0] - (float) 0.5 * hl,
                         this->dcm_.org_ye[this->dcm_.org_dim_.y] + (float) 0.5 * hl };
        R       z[2] = { this->dcm_.org_ze[0] - (float) 0.5 * hl,
                         this->dcm_.org_ze[this->dcm_.org_dim_.z] + (float) 0.5 * hl };

        this->world->geo = new mqi::grid3d<density_t, R>(x, 2, y, 2, z, 2);

        this->world->geo->fill_data(0.0);   //air
        this->world->n_scorers                          = 0;
        this->world->scorers                            = nullptr;
        mqi::beamline<R>            beamline            = this->tx->get_beamline(bnb);
        std::vector<mqi::geometry*> beamline_geometries = beamline.get_geometries();
        this->world->n_children                         = beamline_geometries.size() + 1;
        this->world->children                           = new node_t<R>*[this->world->n_children];
        //
        ///< Create beamline objects
        mqi::coordinate_transform<R> p_coord = this->tx->get_coordinate(bnb);
        p_coord.angles[3]                    = 90.0;   //iec2dicom angle
        node_t<R>** beamline_objects         = new node_t<R>*[beamline_geometries.size()];
        for (int i = 0; i < beamline_geometries.size(); i++) {
            beamline_objects[i] = new node_t<R>;
            if (beamline_geometries[i]->geotype == mqi::RANGESHIFTER) {
                printf("RANGE SHIFTER added\n");
                beamline_objects[i] = this->create_rangeshifter(
                  dynamic_cast<mqi::rangeshifter*>(beamline_geometries[i]), p_coord);
            } else if (beamline_geometries[i]->geotype == mqi::BLOCK) {
                printf("APERTURE added\n");
                beamline_objects[i] = this->create_voxelized_aperture(
                  dynamic_cast<mqi::aperture*>(beamline_geometries[i]));
            }

            beamline_objects[i]->n_scorers = 0;   // Don't score anything on beamline objects
            this->world->children[i]       = beamline_objects[i];
        }

        ///< create a child
        printf("dcm dim %d %d %d\n", this->dcm_.dim_.x, this->dcm_.dim_.y, this->dcm_.dim_.z);
        node_t<R>* phantom                                = new node_t<R>;
        this->world->children[beamline_geometries.size()] = phantom;
        phantom->geo                                      = new grid3d<density_t, R>(this->dcm_.xe,
                                                this->dcm_.dim_.x + 1,
                                                this->dcm_.ye,
                                                this->dcm_.dim_.y + 1,
                                                this->dcm_.ze,
                                                this->dcm_.dim_.z + 1);
        density_t* rho_mass = new density_t[dcm_.dim_.x * dcm_.dim_.y * dcm_.dim_.z];
        for (int i = 0; i < dcm_.dim_.x * dcm_.dim_.y * dcm_.dim_.z; i++) {
            rho_mass[i] =
              this->tx->material_->hu_to_density(this->ct_data[i]) * this->density_scale;
        }
        phantom->geo->set_data(rho_mass);   //// Material conversion function required
        mqi::mask_reader mask_reader0(this->dcm_.dim_);
        mqi::mask_reader mask_reader1(this->dcm_.dim_);
        roi_t*           roi_tmp;
        roi_t*           roi_tmp_stat;

        if (scoring_mask) {
            mask_reader0.mask_filenames = mask_filenames;
            mask_reader0.read_mask_files();
            roi_tmp = mask_reader0.mask_to_roi();
        } else if (this->read_structure) {
            mask_reader0.set_mask(this->dcm_.body_contour);
            roi_tmp = mask_reader0.mask_to_roi();
        } else {
            roi_tmp =
              new roi_t(mqi::DIRECT, this->dcm_.dim_.x * this->dcm_.dim_.y * this->dcm_.dim_.z);
        }

        phantom->n_scorers = 0;
        for (int ss = 0; ss < this->scorer_type.size(); ss++) {
            if (this->scorer_type[ss] == mqi::DOSE || this->scorer_type[ss] == mqi::DOSE_Dij) {
                phantom->n_scorers += 1;
            } else if (this->scorer_type[ss] == mqi::LETd || this->scorer_type[ss] == mqi::LETt) {
                phantom->n_scorers += 2;
            }
        }

        if (this->record_statistics) {
            if (this->set_stat_roi_from_rtstruct) {
                mask_reader1.set_mask(this->dcm_.roi_contour);
                roi_tmp_stat = mask_reader1.mask_to_roi();
            } else if (this->stat_roi_mask_filenames.size() > 0) {
                mask_reader1.mask_filenames = this->stat_roi_mask_filenames;
                mask_reader1.read_mask_files();
                roi_tmp_stat = mask_reader1.mask_to_roi();
            } else {
                printf("!!!!!!!!!!!!!!!!!!!!!!!!! WARNING !!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
                printf("Statistical ROI is set to entire patient.\n");
                printf("This might decrease the performance of the simulation\n");
                printf("Please, considering set an ROI for statistical criteria\n");
                printf("!!!!!!!!!!!!!!!!!!!!!!!!! WARNING !!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
                roi_tmp_stat =
                  new roi_t(mqi::DIRECT, this->dcm_.dim_.x * this->dcm_.dim_.y * this->dcm_.dim_.z);
            }
            this->stat_roi_size = roi_tmp_stat->get_mask_size();
        }

        if (this->record_statistics) {
            phantom->n_scorers += 2;
        }

        phantom->scorers           = new scorer<R>*[phantom->n_scorers];
        fp_compute_hit<R>* fp      = new fp_compute_hit<R>[phantom->n_scorers];
        mqi::key_value**   deposit = new mqi::key_value*[phantom->n_scorers];
        fp_compute_hit<R>  fp_dose, fp_dose_square, fp_letd_weight1, fp_letd_weight2;
        fp_compute_hit<R>  fp_lett_weight1, fp_lett_weight2;

#if defined(__CUDACC__)
        cudaMemcpyFromSymbol(&fp_dose, mqi::Dw_pointer, sizeof(fp_compute_hit<R>));
        cudaMemcpyFromSymbol(&fp_dose_square, mqi::Dw_square_pointer, sizeof(fp_compute_hit<R>));
        cudaMemcpyFromSymbol(
          &fp_letd_weight1, mqi::LETd_weight1_pointer, sizeof(fp_compute_hit<R>));
        cudaMemcpyFromSymbol(
          &fp_letd_weight2, mqi::LETd_weight2_pointer, sizeof(fp_compute_hit<R>));
        cudaMemcpyFromSymbol(
          &fp_lett_weight1, mqi::LETt_weight1_pointer, sizeof(fp_compute_hit<R>));
        cudaMemcpyFromSymbol(
          &fp_lett_weight2, mqi::LETt_weight2_pointer, sizeof(fp_compute_hit<R>));

#else
        fp_dose         = mqi::dose_to_water;
        fp_dose_square  = mqi::dose_to_water_square;
        fp_letd_weight1 = mqi::LETd_weight1;
        fp_letd_weight2 = mqi::LETd_weight2;
        fp_lett_weight1 = mqi::LETt_weight1;
        fp_lett_weight2 = mqi::LETt_weight2;
#endif
        int             current_scorer = 0;
        mqi::key_value *deposit0, *deposit1, *deposit2;

        for (int ss = 0; ss < this->scorer_type.size(); ss++) {
            if (this->scorer_type[ss] == mqi::DOSE) {
                phantom->scorers[current_scorer] =
                  new mqi::scorer<R>(this->scorer_string[ss].c_str(),
                                     this->dcm_.dim_.x * this->dcm_.dim_.y * this->dcm_.dim_.z,
                                     fp_dose);
                deposit0 = new mqi::key_value[phantom->scorers[current_scorer]->max_capacity_];
                std::memset(deposit0,
                            0xff,
                            sizeof(mqi::key_value) *
                              phantom->scorers[current_scorer]->max_capacity_);
                init_table(deposit0, phantom->scorers[current_scorer]->max_capacity_);
                phantom->scorers[current_scorer]->data_ = deposit0;
                phantom->scorers[current_scorer]->roi_  = roi_tmp;
                current_scorer++;
            } else if (this->scorer_type[ss] == mqi::LETd) {
                phantom->scorers[current_scorer] =
                  new mqi::scorer<R>("LETd_numer",
                                     this->dcm_.dim_.x * this->dcm_.dim_.y * this->dcm_.dim_.z,
                                     fp_letd_weight1);

                deposit1 = new mqi::key_value[phantom->scorers[current_scorer]->max_capacity_];
                std::memset(deposit1,
                            0xff,
                            sizeof(mqi::key_value) *
                              phantom->scorers[current_scorer]->max_capacity_);
                init_table(deposit1, phantom->scorers[current_scorer]->max_capacity_);
                phantom->scorers[current_scorer]->data_ = deposit1;
                phantom->scorers[current_scorer]->roi_  = roi_tmp;
                current_scorer++;
                phantom->scorers[current_scorer] =
                  new mqi::scorer<R>("LETd_denom",
                                     this->dcm_.dim_.x * this->dcm_.dim_.y * this->dcm_.dim_.z,
                                     fp_letd_weight2);
                deposit2 = new mqi::key_value[phantom->scorers[current_scorer]->max_capacity_];
                std::memset(deposit2,
                            0xff,
                            sizeof(mqi::key_value) *
                              phantom->scorers[current_scorer]->max_capacity_);
                init_table(deposit2, phantom->scorers[current_scorer]->max_capacity_);
                phantom->scorers[current_scorer]->data_ = deposit2;
                phantom->scorers[current_scorer]->roi_  = roi_tmp;
                current_scorer++;
            } else if (this->scorer_type[ss] == mqi::LETt) {
                phantom->scorers[current_scorer] =
                  new mqi::scorer<R>("LETt_numer",
                                     this->dcm_.dim_.x * this->dcm_.dim_.y * this->dcm_.dim_.z,
                                     fp_lett_weight1);
                deposit1 = new mqi::key_value[phantom->scorers[current_scorer]->max_capacity_];
                std::memset(deposit1,
                            0xff,
                            sizeof(mqi::key_value) *
                              phantom->scorers[current_scorer]->max_capacity_);
                init_table(deposit1, phantom->scorers[current_scorer]->max_capacity_);
                phantom->scorers[current_scorer]->data_ = deposit1;
                phantom->scorers[current_scorer]->roi_  = roi_tmp;
                current_scorer++;
                phantom->scorers[current_scorer] =
                  new mqi::scorer<R>("LETt_denom",
                                     this->dcm_.dim_.x * this->dcm_.dim_.y * this->dcm_.dim_.z,
                                     fp_lett_weight2);
                deposit2 = new mqi::key_value[phantom->scorers[current_scorer]->max_capacity_];
                std::memset(deposit2,
                            0xff,
                            sizeof(mqi::key_value) *
                              phantom->scorers[current_scorer]->max_capacity_);
                init_table(deposit2, phantom->scorers[current_scorer]->max_capacity_);
                phantom->scorers[current_scorer]->data_ = deposit2;
                phantom->scorers[current_scorer]->roi_  = roi_tmp;
                current_scorer++;
            } else if (this->scorer_type[ss] == mqi::DOSE_Dij) {
                phantom->scorers[current_scorer] =
                  new mqi::scorer<R>(this->scorer_string[ss].c_str(), 512 * 512 * 300 * 5, fp_dose);
                deposit0 = new mqi::key_value[phantom->scorers[current_scorer]->max_capacity_];
                std::memset(deposit0,
                            0xff,
                            sizeof(mqi::key_value) *
                              phantom->scorers[current_scorer]->max_capacity_);
                init_table(deposit0, phantom->scorers[current_scorer]->max_capacity_);
                phantom->scorers[current_scorer]->data_ = deposit0;
                phantom->scorers[current_scorer]->roi_  = roi_tmp;
                current_scorer++;
            }
        }
        printf("total scorer %d current scorer %d\n", phantom->n_scorers, current_scorer);

        if (this->record_statistics) {
            phantom->scorers[current_scorer] =
              new mqi::scorer<R>("Dose_stat", this->stat_roi_size, fp_dose, this->save_statistics);
            mqi::key_value* deposit3 =
              new mqi::key_value[phantom->scorers[current_scorer]->max_capacity_];
            std::memset(deposit3,
                        0xff,
                        sizeof(mqi::key_value) * phantom->scorers[current_scorer]->max_capacity_);
            init_table(deposit3, phantom->scorers[current_scorer]->max_capacity_);
            phantom->scorers[current_scorer]->data_ = deposit3;
            phantom->scorers[current_scorer]->roi_  = roi_tmp_stat;
            current_scorer++;
            printf("total scorer %d current scorer %d\n", phantom->n_scorers, current_scorer);
            phantom->scorers[current_scorer] = new mqi::scorer<R>(
              "DoseSquare_stat", this->stat_roi_size, fp_dose_square, this->save_statistics);
            mqi::key_value* deposit4 =
              new mqi::key_value[phantom->scorers[current_scorer]->max_capacity_];
            std::memset(deposit4,
                        0xff,
                        sizeof(mqi::key_value) * phantom->scorers[current_scorer]->max_capacity_);
            init_table(deposit4, phantom->scorers[current_scorer]->max_capacity_);
            phantom->scorers[current_scorer]->data_ = deposit4;
            phantom->scorers[current_scorer]->roi_  = roi_tmp_stat;
            current_scorer++;
        }
        //        mc::mc_score_variance = this->score_variance;
    }

    CUDA_HOST
    virtual void
    setup_beamsource() {
        uint16_t                 num_beams  = this->tx->get_num_beams();
        std::vector<std::string> beam_names = this->tx->get_beam_names();
        printf("There are %d beams\n", num_beams);
        printf("Selecting %d: %s\n", bnb, beam_names[bnb - 1].c_str());
        const mqi::dataset* mqi_ds            = this->tx->get_beam_dataset(bnb);
        const mqi::dataset* ion_beam_sequence = (*mqi_ds)("IonControlPointSequence")[0];
        std::vector<float>  temp_sid;
        ion_beam_sequence->get_values("SnoutPosition", temp_sid);
        this->sid = temp_sid[0] + 50;
        printf("sid %f\n", this->sid);
        mqi::coordinate_transform<R> p_coord = this->tx->get_coordinate(bnb);
        p_coord.angles[3]                    = 90.0;   //iec2dicom angle
        printf("p_coord %f %f %f %f\n",
               p_coord.angles[0],
               p_coord.angles[1],
               p_coord.angles[2],
               p_coord.angles[3]);
        p_coord.translation = p_coord.translation;   // - this->dcm_.image_center;
        mqi::coordinate_transform<R> p_final(p_coord.angles, p_coord.translation);
        printf("translation");
        p_final.translation.dump();
        printf("rotation ");
        p_final.rotation.dump();
        this->beamsource = this->tx->get_beamsource(bnb, p_final, this->particles_per_history, sid);
        printf("beamlets %lu\n", this->beamsource.total_beamlets());
        printf("histories %lu\n", this->beamsource.total_histories());
    }

    CUDA_HOST
    void
    initialize_and_run() {
        for (int beam_queue = 0; beam_queue < beam_numbers.size(); beam_queue++) {
            this->bnb = beam_numbers[beam_queue];
            this->master_seed += beam_queue * 10000;
            this->beam_rng.seed(this->master_seed);
            printf("bnb %d seed %d\n", this->bnb, this->master_seed);
            this->initialize();
            this->run();
            this->finalize();
            if (this->reshape_output) {
                this->save_reshaped_files();
            } else if (this->sparse_output) {
                this->save_sparse_file();
            }
        }
    }
    CUDA_HOST
    virtual void
    run() {
        size_t free, total;
        printf("scorer %d sim type %d\n", scorer_type, sim_type);
        if (this->sim_type == mqi::PER_BEAM) {
#if defined(__CUDACC__)
            cudaMemGetInfo(&free, &total);
            printf("Geometry occupies %f GB\n", (total - free) / (1024.0 * 1024.0 * 1024.0));
#endif
            if (this->record_statistics) {
                run_by_beam_stat();
            } else {
                run_by_beam();
            }
        } else if (this->sim_type == mqi::PER_SPOT) {
#if defined(__CUDACC__)
            cudaMemGetInfo(&free, &total);
            printf("Geometry occupies %f GB\n", (total - free) / (1024.0 * 1024.0 * 1024.0));
#endif
            run_by_spot();
        }

    }   // run

    CUDA_HOST
    virtual void
    run_simulation(size_t    histories_per_batch,
                   size_t    histories_in_batch,
                   uint32_t* tracked_particles,
                   int32_t*  transport_seed,
                   uint32_t* scorer_offset_vector = nullptr) {
        /// histories_per_batch and histories_in_batch are kine of redundant.
        /// the histories_per_batch may not required if copying memory work correctly with histories_in_batch
        auto start = std::chrono::high_resolution_clock::now();
        // TODO: divide vertices into sub-vertices if it is too large
        uint32_t                                  n_threads            = 0;
        uint32_t                                  n_blocks             = 0;
        uint32_t                                  particles_per_thread = 0;
        mqi::thrd_t*                              worker_threads;
        auto                                      stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = stop - start;
        printf("Initialization for run done %f s\n", duration.count() * 0.0001);
#if defined(__CUDACC__)
        start     = std::chrono::high_resolution_clock::now();
        n_threads = thread_limit;
        if (histories_in_batch < n_threads) {
            n_threads            = histories_in_batch;
            n_blocks             = 1;
            particles_per_thread = histories_in_batch / n_threads;
        } else if (this->num_total_threads > 0) {
            if (this->num_total_threads <= thread_limit) {
                n_threads = this->num_total_threads;
                n_blocks  = 1;
            } else {
                n_threads = thread_limit;
                n_blocks  = (int) std::ceil(this->num_total_threads * 1.0 / n_threads);
                if (n_blocks > block_limit) {
                    n_blocks = n_blocks;
                }
            }
        } else if (this->num_total_threads < 0) {
            n_threads = n_threads;
            n_blocks  = (int) std::ceil(histories_in_batch * 1.0 / n_threads);
            if (n_blocks > block_limit)   //maybe larger?
                n_blocks = block_limit;
            particles_per_thread =
              (int) std::ceil(histories_in_batch * 1.0 / (n_threads * n_blocks));
            if (histories_in_batch % (n_threads * n_blocks * particles_per_thread) > 0)
                n_blocks += 1;   // increase block size if there is any remainder
        }
        printf("Block size: %d, Thread size: %d\n", n_blocks, n_threads);
        mc::upload_vertices(this->vertices, mc::mc_vertices, 0, histories_per_batch);
        cudaDeviceSynchronize();
        check_cuda_last_error("(upload vertices)");
        uint32_t* d_scorer_offset_vector;
        if (scorer_offset_vector) {
            printf("Histories per batch %d\n", histories_per_batch);
            mc::upload_scorer_offset_vector(
              scorer_offset_vector, d_scorer_offset_vector, histories_per_batch);
            cudaDeviceSynchronize();
        } else {
            d_scorer_offset_vector = nullptr;
        }
        uint32_t* d_tracked_particles;
        gpu_err_chk(cudaMalloc(&worker_threads, n_blocks * n_threads * sizeof(mqi::thrd_t)));

        int32_t* d_transport_seed;
        gpu_err_chk(cudaMalloc(&d_transport_seed, sizeof(int32_t) * histories_per_batch));
        gpu_err_chk(cudaMemcpy(d_transport_seed,
                               transport_seed,
                               sizeof(int32_t) * histories_per_batch,
                               cudaMemcpyHostToDevice));
        start = std::chrono::high_resolution_clock::now();
        initialize_threads<<<n_blocks, n_threads>>>(
          worker_threads, n_blocks * n_threads, this->master_seed);
        cudaDeviceSynchronize();
        check_cuda_last_error("(initialize threads)");
        gpu_err_chk(cudaMalloc(&d_tracked_particles, sizeof(tracked_particles[0])));
        gpu_err_chk(cudaMemcpy(d_tracked_particles,
                               tracked_particles,
                               sizeof(tracked_particles[0]),
                               cudaMemcpyHostToDevice));

        printf("Simulating particles...\n");
        if (this->record_statistics) {
            mc::transport_particles_patient_stat<R>
              <<<n_blocks, n_threads>>>(worker_threads,
                                        mc::mc_world,
                                        mc::mc_vertices,
                                        mc::mc_materials,
                                        histories_in_batch,
                                        d_tracked_particles,
                                        d_transport_seed,
                                        d_scorer_offset_vector);
        } else {
            mc::transport_particles_patient<R><<<n_blocks, n_threads>>>(worker_threads,
                                                                        mc::mc_world,
                                                                        mc::mc_vertices,
                                                                        mc::mc_materials,
                                                                        histories_in_batch,
                                                                        d_tracked_particles,
                                                                        d_transport_seed,
                                                                        d_scorer_offset_vector);
        }

        cudaDeviceSynchronize();
        check_cuda_last_error("(transport particle)");
        gpu_err_chk(cudaMemcpy(tracked_particles,
                               d_tracked_particles,
                               sizeof(tracked_particles[0]),
                               cudaMemcpyDeviceToHost));
        gpu_err_chk(cudaFree(d_tracked_particles));
        gpu_err_chk(cudaFree(worker_threads));
        gpu_err_chk(cudaFree(mc::mc_vertices));
        gpu_err_chk(cudaFree(d_transport_seed));
#else
        n_threads       = 1;
        mc::mc_vertices = this->vertices;
        mc::mc_world    = this->world;
        /// TODO: number of threads for multithreading implementation
        worker_threads = new mqi::thrd_t[n_threads];
        initialize_threads(worker_threads, n_threads, this->master_seed);
        mc::transport_particles_patient<R>(
          worker_threads, mc::mc_world, mc::mc_vertices, histories_in_batch, tracked_particles);
#endif

    }   //run_simulation

    CUDA_HOST
    virtual void
    run_by_beam(mqi::node_t<R>* world = mc::mc_world) {
        //// Beam simulation
        /// TODO: faster implementation
        printf("Run by beam\n");
        std::chrono::time_point<std::chrono::high_resolution_clock> start, stop;
        std::chrono::duration<double, std::milli>                   duration;
        size_t                                                      h0 = 0;
        size_t    h1                = this->beamsource.total_histories();
        uint32_t  num_vertices      = h1 - h0;
        uint32_t* tracked_particles = new uint32_t[1];
        tracked_particles[0]        = 0;
        int num_batches;
        int spot_id             = 0;
        int accum_histories     = std::get<2>(this->beamsource[spot_id]);
        int accum_histories_tmp = std::get<2>(this->beamsource[spot_id]);
        this->beam_rng.seed(this->master_seed + spot_id);
        size_t histories_per_batch = 0, cum_vertices = 0;
        size_t current_vertex = 0;
        if (this->max_histories_per_batch <= 0) {
            num_batches         = 1;
            histories_per_batch = (h1 - h0);   // upload all vertices at onc
            printf("Uploading %lu histories\n", histories_per_batch);
        } else {
            num_batches = (int) mqi::mqi_ceil((h1 - h0) * 1.0 / this->max_histories_per_batch);
            histories_per_batch = this->max_histories_per_batch;
            printf("Upload %lu histories per batch, %d batches expected\n",
                   histories_per_batch,
                   num_batches);
        }

        ////Transportation random seed
        std::default_random_engine tmp_rng;
        int32_t*                   transport_seed = new int32_t[histories_per_batch];
        ////Transportation random seed
        for (int batch = 0; batch < num_batches; batch++) {
            start          = std::chrono::high_resolution_clock::now();
            this->vertices = new mqi::vertex_t<R>[histories_per_batch];
            printf("Generating particles...\n");
            for (current_vertex = 0; current_vertex < histories_per_batch; current_vertex++) {
                if (cum_vertices + current_vertex >= (h1 - h0)) {
                    /// If reach the final vertex
                    break;
                }
                if (cum_vertices + current_vertex >= accum_histories) {
                    spot_id += 1;
                    while (std::get<2>(this->beamsource[spot_id]) == accum_histories) {
                        spot_id += 1;
                    }
                    accum_histories = std::get<2>(this->beamsource[spot_id]);
                    this->beam_rng.seed(this->master_seed + spot_id);
                    tmp_rng.seed(this->master_seed + spot_id);
                }
                transport_seed[current_vertex] = tmp_rng();
                auto bl                        = this->beamsource(cum_vertices + current_vertex);
                this->vertices[current_vertex] = bl(&this->beam_rng);
            }
            stop     = std::chrono::high_resolution_clock::now();
            duration = stop - start;
            printf("Beam generation %f s\n", duration.count() * 0.0001);
            cum_vertices += current_vertex;
            printf("cum_vertiex %lu %lu\n", cum_vertices, current_vertex);
            /// Transport particles
            start = std::chrono::high_resolution_clock::now();
            printf("Transporting particles...\n");
            run_simulation(histories_per_batch, current_vertex, tracked_particles, transport_seed);
            stop     = std::chrono::high_resolution_clock::now();
            duration = stop - start;
            printf("Run transportation %f s\n", duration.count() * 0.0001);
            delete[] this->vertices;
            if (tracked_particles[0] == h1) {
                break;
            }
        }
        printf("cum vertex %lu total histories %lu\n", cum_vertices, h1);
        printf("Number of particles tracked %d\n", tracked_particles[0]);

    }   //run_by_beam
    CUDA_HOST
    virtual void
    run_by_beam_stat(mqi::node_t<R>* world = mc::mc_world) {
        //// Beam simulation
        /// TODO: faster implementation
        float current_stat = 100.0;
        printf("Run by beam\n");
        std::chrono::time_point<std::chrono::high_resolution_clock> start, stop;
        std::chrono::duration<double, std::milli>                   duration;
        size_t                                                      h0 = 0;
        this->num_spots                                                = 1;
        size_t    h1                = this->beamsource.total_histories();
        uint32_t  num_vertices      = h1 - h0;
        uint32_t* tracked_particles = new uint32_t[1];
        tracked_particles[0]        = 0;
        int    num_batches;
        int    spot_id             = 0;
        int    accum_histories     = std::get<2>(this->beamsource[spot_id]);
        int    accum_histories_tmp = std::get<2>(this->beamsource[spot_id]);
        size_t histories_per_batch = 0, cum_vertices = 0;
        size_t current_vertex = 0;
        int    count_run      = 0;
        while (current_stat > stat_criteria) {
            if (current_stat == 100.0) {
                printf("Running %d histories for the first run\n", num_vertices);
            } else {
                printf("Running additional %d histories\n", histories_per_batch);
            }
            count_run += 1;
            histories_per_batch = 0;
            cum_vertices        = 0;
            current_vertex      = 0;
            this->beam_rng.seed(this->master_seed + spot_id + 1);
            if (this->max_histories_per_batch <= 0) {
                num_batches         = 1;
                histories_per_batch = (h1 - h0);   // upload all vertices at onc
                printf("Uploading %lu histories\n", histories_per_batch);
            } else {
                num_batches = (int) mqi::mqi_ceil((h1 - h0) * 1.0 / this->max_histories_per_batch);
                histories_per_batch = this->max_histories_per_batch;
                printf("Upload %lu histories per batch, %d batches expected\n",
                       histories_per_batch,
                       num_batches);
            }
            ////Transportation random seed
            std::default_random_engine tmp_rng;
            int32_t*                   transport_seed = new int32_t[histories_per_batch];
            ////Transportation random seed

            for (int batch = 0; batch < num_batches; batch++) {
                start          = std::chrono::high_resolution_clock::now();
                this->vertices = new mqi::vertex_t<R>[histories_per_batch];
                printf("Generating particles...\n");

                for (current_vertex = 0; current_vertex < histories_per_batch; current_vertex++) {
                    if (cum_vertices + current_vertex >= (h1 - h0)) {
                        /// If reach the final vertex
                        break;
                    }
                    if (cum_vertices + current_vertex >= accum_histories) {
                        spot_id += 1;
                        while (std::get<2>(this->beamsource[spot_id]) == accum_histories) {
                            spot_id += 1;
                        }
                        accum_histories = std::get<2>(this->beamsource[spot_id]);
                        this->beam_rng.seed(this->master_seed + spot_id);
                        tmp_rng.seed(this->master_seed + spot_id);
                    }
                    transport_seed[current_vertex] = tmp_rng();
                    auto bl = this->beamsource(cum_vertices + current_vertex);
                    this->vertices[current_vertex] = bl(&this->beam_rng);
                }
                stop     = std::chrono::high_resolution_clock::now();
                duration = stop - start;
                printf("Beam generation %f s\n", duration.count() * 0.0001);
                cum_vertices += current_vertex;
                printf("Cumulated vertices %lu %lu\n", cum_vertices, current_vertex);
                /// Transport particles
                start = std::chrono::high_resolution_clock::now();
                printf("Transporting particles...\n");
                run_simulation(
                  histories_per_batch, current_vertex, tracked_particles, transport_seed);
                stop     = std::chrono::high_resolution_clock::now();
                duration = stop - start;
                printf("Run transportation %f s\n", duration.count() * 0.0001);
                delete[] this->vertices;
                if (tracked_particles[0] == h1) {
                    break;
                }
            }
            current_stat = calculate_stat(tracked_particles[0]) * 100.0;
            printf("Number of particles tracked %d\n", tracked_particles[0]);
            printf("Run %d: current uncertainty %f %\n", count_run, current_stat);
        }
        calculate_average_results(tracked_particles[0], h1 - h0);
    }   //run_by_beam_stat

    CUDA_HOST
    float
    calculate_stat(int n_histories) {
        auto                                      start = std::chrono::high_resolution_clock::now();
        auto                                      stop  = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = stop - start;
        float*   standard_deviation                        = new float[this->stat_roi_size];
        float*   dose_mean                                 = new float[this->stat_roi_size];
        uint32_t n_threads                                 = 0;
        uint32_t n_blocks                                  = 0;
        uint32_t voxels_per_thread                         = 0;
        for (int i = 0; i < this->stat_roi_size; i++) {
            standard_deviation[i] = 0;
        }
#if defined(__CUDACC__)
        n_threads = thread_limit;
        if (this->stat_roi_size < n_threads) {
            n_threads         = this->stat_roi_size;
            n_blocks          = 1;
            voxels_per_thread = this->stat_roi_size / n_threads;
        } else if (this->num_total_threads > 0) {
            if (this->num_total_threads <= thread_limit) {
                n_threads = this->num_total_threads;
                n_blocks  = 1;
            } else {
                n_threads = thread_limit;
                n_blocks  = (int) std::ceil(this->num_total_threads * 1.0 / n_threads);
                if (n_blocks > block_limit) {
                    n_blocks = n_blocks;
                }
            }
        } else if (this->num_total_threads < 0) {
            n_threads = n_threads;
            n_blocks  = (int) std::ceil(this->stat_roi_size * 1.0 / n_threads);
            if (n_blocks > block_limit)   //maybe larger?
                n_blocks = block_limit;
            voxels_per_thread = (int) std::ceil(this->stat_roi_size * 1.0 / (n_threads * n_blocks));
            if (this->stat_roi_size % (n_threads * n_blocks * voxels_per_thread) > 0)
                n_blocks += 1;   // increase block size if there is any remainder
        }
        float* d_standard_deviation;
        float* d_dose_mean;
        gpu_err_chk(cudaMalloc(&d_standard_deviation, sizeof(float) * this->stat_roi_size));
        gpu_err_chk(cudaMemcpy(d_standard_deviation,
                               standard_deviation,
                               sizeof(float) * this->stat_roi_size,
                               cudaMemcpyHostToDevice));
        gpu_err_chk(cudaMalloc(&d_dose_mean, sizeof(float) * this->stat_roi_size));
        gpu_err_chk(cudaMemcpy(
          d_dose_mean, dose_mean, sizeof(float) * this->stat_roi_size, cudaMemcpyHostToDevice));
        printf("Calculating standard deviation\n");
        calculate_standard_deviation<R><<<n_blocks, n_threads>>>(mc::mc_world,
                                                                 d_standard_deviation,
                                                                 d_dose_mean,
                                                                 n_histories,
                                                                 this->stat_roi_size,
                                                                 this->world->n_children - 1);
        cudaDeviceSynchronize();
        check_cuda_last_error("(calculate_standard_deviation)");
        printf("Standard deviation calculation done\n");
        gpu_err_chk(cudaMemcpy(standard_deviation,
                               d_standard_deviation,
                               sizeof(float) * this->stat_roi_size,
                               cudaMemcpyDeviceToHost));
        gpu_err_chk(cudaFree(d_standard_deviation));
        gpu_err_chk(cudaMemcpy(
          dose_mean, d_dose_mean, sizeof(float) * this->stat_roi_size, cudaMemcpyDeviceToHost));
        gpu_err_chk(cudaFree(d_dose_mean));
#else

#endif
        double current_stat = 0;
        double dose_max     = 0;
        int    count        = 0;
        for (int i = 0; i < this->stat_roi_size; i++) {
            if (dose_max < dose_mean[i]) {
                dose_max = dose_mean[i];
            }
        }
        for (int i = 0; i < this->stat_roi_size; i++) {
            if (dose_mean[i] > dose_max * this->stat_threshold) {
                //                current_stat += (standard_deviation[i].value / dose_mean[i].value);
                current_stat += (standard_deviation[i] / dose_mean[i]);
                count += 1;
            }
        }
        current_stat /= count;
        return current_stat;
    }   //calculate_stat

    CUDA_HOST
    void
    calculate_average_results(int n_histories, int target_histories) {
        auto                                      start = std::chrono::high_resolution_clock::now();
        auto                                      stop  = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration  = stop - start;
        float*                                    dose_mean = new float[this->stat_roi_size];
        uint32_t                                  n_threads = 0;
        uint32_t                                  n_blocks  = 0;
        uint32_t                                  voxels_per_thread = 0;
        float                                     weight = target_histories * 1.0 / n_histories;
#if defined(__CUDACC__)
        n_threads = thread_limit;
        if (this->stat_roi_size < n_threads) {
            n_threads         = this->stat_roi_size;
            n_blocks          = 1;
            voxels_per_thread = this->stat_roi_size / n_threads;
        } else if (this->num_total_threads > 0) {
            if (this->num_total_threads <= thread_limit) {
                n_threads = this->num_total_threads;
                n_blocks  = 1;
            } else {
                n_threads = thread_limit;
                n_blocks  = (int) std::ceil(this->num_total_threads * 1.0 / n_threads);
                if (n_blocks > block_limit) {
                    n_blocks = n_blocks;
                }
            }
        } else if (this->num_total_threads < 0) {
            n_threads = n_threads;
            n_blocks  = (int) std::ceil(this->stat_roi_size * 1.0 / n_threads);
            if (n_blocks > block_limit)   //maybe larger?
                n_blocks = block_limit;
            voxels_per_thread = (int) std::ceil(this->stat_roi_size * 1.0 / (n_threads * n_blocks));
            if (this->stat_roi_size % (n_threads * n_blocks * voxels_per_thread) > 0)
                n_blocks += 1;   // increase block size if there is any remainder
        }
        //        printf("Block size: %d, Thread size: %d\n", n_blocks, n_threads);
        calculate_average<R>
          <<<n_blocks, n_threads>>>(mc::mc_world,
                                    weight,
                                    this->dcm_.dim_.x * this->dcm_.dim_.y * this->dcm_.dim_.z,
                                    this->world->n_children - 1);
        cudaDeviceSynchronize();
        check_cuda_last_error("(calculate_average)");
#else

#endif
    }   // calculate_average_results
    CUDA_HOST
    virtual void
    run_by_spot(mqi::node_t<R>* world = mc::mc_world) {
        //// Spot by spot simulation
        /// TODO: faster implementation
        printf("Run by spot\n");
        std::chrono::time_point<std::chrono::high_resolution_clock> start, stop;
        std::chrono::duration<double, std::milli>                   duration;
        size_t                                                      h0 = 0;
        size_t h1       = this->beamsource.total_histories();
        this->num_spots = this->beamsource.total_beamlets();
        if (this->unit_weights > 0) {
            h1                          = this->num_spots * this->unit_weights;
            this->particles_per_history = 1;
        }
        size_t    max_histories     = 0;
        uint32_t* tracked_particles = new uint32_t[1];
        tracked_particles[0]        = 0;
        printf("num spots %d\n", this->num_spots);
        size_t    total_history   = 0;
        uint32_t* spot_boundaries = new uint32_t[this->num_spots];
        size_t    num_histories, cum_histories = 0;
        size_t    num_batches, history_ind, loop_end;
        size_t    cum_vertices = 0, spot_start = 0, batch = 0, spot_ind = 0;
        size_t    current_history     = 0;   // Number of histories from current spot id
        size_t    current_vertex      = 0;   // Number of vertex in current batch
        size_t    histories_per_batch = 0;

        if (this->max_histories_per_batch == 0) {
            num_batches         = 1;
            histories_per_batch = (h1 - h0);   // upload all vertices at onc
            printf("Upload %lu histories\n", histories_per_batch);
        } else {
            num_batches         = (int) mqi::mqi_ceil(h1 * 1.0 / this->max_histories_per_batch);
            histories_per_batch = this->max_histories_per_batch;
            printf("Upload %lu histories per batch, %lu batches expected\n",
                   histories_per_batch,
                   num_batches);
        }

        mqi::vertex_t<R>* vertices_test;
        ////Transportation random seed
        std::default_random_engine tmp_rng;
        int32_t*                   transport_seed = new int32_t[histories_per_batch];
        ////Transportation random seed

        while (spot_ind < this->num_spots) {
            this->vertices                = new mqi::vertex_t<R>[histories_per_batch];
            vertices_test                 = new mqi::vertex_t<R>[histories_per_batch];
            uint32_t* score_offset_vector = new uint32_t[histories_per_batch];
            //            printf("num batches %d batch %d spot start %d\n",num_batches,batch, spot_start);
            start = std::chrono::high_resolution_clock::now();
            printf("Generating particles...\n");
            for (spot_ind = spot_start; spot_ind < this->num_spots; spot_ind++) {
                auto bl = this->beamsource[spot_ind];
                if (this->unit_weights > 0) {
                    num_histories = this->unit_weights;
                } else {
                    num_histories = std::get<1>(bl);
                }
                if (current_history == 0) {
                    this->beam_rng.seed(this->master_seed + spot_ind);
                    tmp_rng.seed(this->master_seed + spot_ind);
                }
                if (num_histories == 0) continue;
                if (num_histories - current_history < histories_per_batch - current_vertex) {
                    loop_end = num_histories - current_history + current_vertex;
                } else {
                    loop_end = histories_per_batch;
                }
                for (int history_ind = current_vertex; history_ind < loop_end; history_ind++) {
                    this->vertices[history_ind]      = std::get<0>(bl)(&this->beam_rng);
                    score_offset_vector[history_ind] = spot_ind;
                    transport_seed[history_ind]      = tmp_rng();
                    assert(history_ind < histories_per_batch);
                }
                /// The multithreading gives small performance gain if we need run it for each spot
                ///20 seconds ->  18 seconds
                assert(loop_end > current_vertex);
                current_history += loop_end - current_vertex;
                cum_vertices += loop_end - current_vertex;
                current_vertex += loop_end - current_vertex;

                if (current_vertex == histories_per_batch && current_history < num_histories) {
                    /// The spot have remaining particles to simulate
                    spot_start = spot_ind;
                    break;
                } else if (current_vertex == histories_per_batch &&
                           current_history == num_histories) {
                    /// The spot have no remaining particles to simulate
                    spot_start = spot_ind + 1;
                    break;
                } else {
                    current_history = 0;
                }
            }
            stop     = std::chrono::high_resolution_clock::now();
            duration = stop - start;

            printf("beam generation %f s\n", duration.count() * 0.0001);
            /// Transport particles
            printf("Transporting particles...\n");
            start = std::chrono::high_resolution_clock::now();
            run_simulation(histories_per_batch,
                           current_vertex,
                           tracked_particles,
                           transport_seed,
                           score_offset_vector);
            stop     = std::chrono::high_resolution_clock::now();
            duration = stop - start;
            printf("run simulation %f s\n", duration.count() * 0.0001);
            current_vertex = 0;
            delete[] this->vertices;
            delete[] score_offset_vector;
            batch += 1;
            if (tracked_particles[0] == h1) {
                break;
            }
        }
        printf("spot ind %lu num_spots %d cum vertices %lu total histories %lu\n",
               spot_ind,
               this->num_spots,
               cum_vertices,
               h1);
        printf("Number of particles tracked %d\n", tracked_particles[0]);

    }   // run_by_spot

    virtual mqi::node_t<R>*
    create_rangeshifter(mqi::rangeshifter* geometry, mqi::coordinate_transform<R> p_coord) {
        mqi::node_t<R>* rangeshifter = new mqi::node_t<R>;
        rangeshifter->n_scorers      = 0;
        rangeshifter->n_children     = 0;
        rangeshifter->scorers        = nullptr;
        rangeshifter->children       = nullptr;
        p_coord.angles[3]            = 90.0;
        printf("p_coord %f %f %f %f\n",
               p_coord.angles[0],
               p_coord.angles[1],
               p_coord.angles[2],
               p_coord.angles[3]);
        mqi::coordinate_transform<R> p_final(p_coord.angles, p_coord.translation);

        printf("RANGESHIFTER\n");
        printf("pos ");
        geometry->pos.dump();
        printf("volume  ");
        geometry->volume.dump();
        printf("create range shifter grid\n");
        rangeshifter->geo = new grid3d<mqi::density_t, R>(geometry->pos.x - geometry->volume.x / 2,
                                                          geometry->pos.x + geometry->volume.x / 2,
                                                          2,
                                                          geometry->pos.y - geometry->volume.y / 2,
                                                          geometry->pos.y + geometry->volume.y / 2,
                                                          2,
                                                          geometry->pos.z - geometry->volume.z / 2,
                                                          geometry->pos.z + geometry->volume.z / 2,
                                                          2,
                                                          p_final.rotation);

        printf("filling range shifter grid\n");
        /// TODO: Check material and density of rangeshifter
        printf("Rangeshifter density %.4f\n", this->rangeshifter_density * 1e-3);
        rangeshifter->geo->fill_data(this->rangeshifter_density * 1e-3);
        rangeshifter->geo->translation_vector = p_final.translation;
        printf("rotation\n");
        p_final.rotation.dump();
        printf("ragneshifter rot \n");
        rangeshifter->geo->rotation_matrix_fwd.dump();
        printf("creating rangeshifter done\n");

        return rangeshifter;
    }

    mqi::node_t<R>*
    create_voxelized_aperture(mqi::aperture* geometry) {
        mqi::node_t<R>*                   aperture = new mqi::node_t<R>;
        std::vector<std::array<float, 2>> block    = geometry->block_data[0];
        geometry->volume.dump();
        geometry->pos.dump();

        uint16_t       num_opening   = geometry->block_data.size();
        uint16_t*      num_segments  = new uint16_t[num_opening];
        mqi::vec2<R>** block_segment = new mqi::vec2<R>*[geometry->block_data.size()];
        for (int i = 0; i < geometry->block_data.size(); i++) {
            block            = geometry->block_data[i];
            block_segment[i] = new mqi::vec2<R>[block.size()];
            num_segments[i]  = block.size();
            for (int j = 0; j < block.size(); j++) {
                std::array<float, 2> pos = block[j];
                block_segment[i][j].x    = pos[0] + geometry->pos.x;
                block_segment[i][j].y    = pos[1] + geometry->pos.y;
            }
        }

        mqi::coordinate_transform<R> p_coord = this->tx->get_coordinate(bnb);
        p_coord.angles[3]                    = 90.0;   //iec2dicom angle
        mqi::vec3<R>      aperture_voxel;
        mqi::vec3<size_t> aperture_dim;
        aperture_voxel.x = 1;
        aperture_voxel.y = 1;
        aperture_voxel.z = 1;
        /// number of voxels
        aperture_dim.x =
          static_cast<size_t>(mqi::mqi_ceil((geometry->volume.x) / aperture_voxel.x));
        aperture_dim.y =
          static_cast<size_t>(mqi::mqi_ceil((geometry->volume.y) / aperture_voxel.y));
        aperture_dim.z =
          static_cast<size_t>(mqi::mqi_ceil((geometry->volume.z) / aperture_voxel.z));
        R* xe = new R[aperture_dim.x + 1];
        R* ye = new R[aperture_dim.y + 1];
        R* ze = new R[aperture_dim.z + 1];
        for (int i = 0; i < aperture_dim.x + 1; i++) {
            xe[i] = (geometry->pos.x - geometry->volume.x / 2) + i * aperture_voxel.x;
        }
        for (int i = 0; i < aperture_dim.y + 1; i++) {
            ye[i] = (geometry->pos.y - geometry->volume.y / 2) + i * aperture_voxel.y;
        }
        for (int i = 0; i < aperture_dim.z + 1; i++) {
            ze[i] = (geometry->pos.z - geometry->volume.z / 2) + i * aperture_voxel.z;
        }
        aperture->n_scorers  = 0;
        aperture->n_children = 0;
        aperture->scorers    = nullptr;
        aperture->children   = nullptr;
        mqi::coordinate_transform<R> p_final(p_coord.angles, p_coord.translation);
        aperture->geo = new mqi::grid3d<mqi::density_t, R>(
          xe, aperture_dim.x + 1, ye, aperture_dim.y + 1, ze, aperture_dim.z + 1, p_final.rotation);
        mqi::density_t* data = new mqi::density_t[aperture_dim.x * aperture_dim.y * aperture_dim.z];
        R               x_pos, y_pos, z_pos;
        size_t          ind;
        mqi::vec3<R>    pos;
        pos.x = 0;
        pos.y = 0;
        pos.z = 0;
        for (int i = 0; i < aperture_dim.x; i++) {
            x_pos = xe[i] + aperture_voxel.x * 0.5;
            pos.x = x_pos;
            for (int j = 0; j < aperture_dim.y; j++) {
                y_pos = ye[j] + aperture_voxel.y * 0.5;
                pos.y = y_pos;
                if (is_inside(pos, num_opening, num_segments, block_segment)) {
                    for (int k = 0; k < aperture_dim.z; k++) {
                        ind = k * aperture_dim.x * aperture_dim.y + j * aperture_dim.x + i;
                        assert(ind < aperture_dim.x * aperture_dim.y * aperture_dim.z);
                        data[ind] = 1e-8;   // open region has low density
                    }
                } else {
                    for (int k = 0; k < aperture_dim.z; k++) {
                        ind = k * aperture_dim.x * aperture_dim.y + j * aperture_dim.x + i;
                        assert(ind < aperture_dim.x * aperture_dim.y * aperture_dim.z);
                        data[ind] = 100.0;   // open region has high density
                    }
                }
            }
        }
        aperture->geo->set_data(data);
        aperture->geo->translation_vector = p_final.translation;
        return aperture;
    }

    CUDA_HOST_DEVICE
    bool
    is_inside(mqi::vec3<R>   pos,
              uint16_t       num_opening,
              uint16_t*      num_segments,
              mqi::vec2<R>** block_segment) {
        bool inside = false;
        for (int i = 0; i < num_opening; i++) {
            mqi::vec2<R>* segment = block_segment[i];
            inside                = sol1_1(pos, segment, num_segments[i]);
        }
        return inside;
    }

    CUDA_HOST_DEVICE
    bool
    sol1_1(mqi::vec3<R> pos, mqi::vec2<R>* segment, uint16_t num_segment) {
        mqi::vec2<R> pos0 = segment[0];
        mqi::vec2<R> pos1;
        float        min_y, max_y, max_x, intersect_x;
        int          count = 0;
        int          i, j, c = 0;
        for (i = 0, j = num_segment - 1; i < num_segment; j = i++) {
            pos0 = segment[i];
            pos1 = segment[j];
            if ((((pos0.y <= pos.y) && (pos.y < pos1.y)) ||
                 ((pos1.y <= pos.y) && (pos.y < pos0.y))) &&
                (pos.x < (pos1.x - pos0.x) * (pos.y - pos0.y) / (pos1.y - pos0.y) + pos0.x)) {
                c = !c;
            }
        }
        return c;
    }
    CUDA_HOST_DEVICE
    bool
    sol1_1(mqi::vec2<float> pos, mqi::vec3<float>*& contour_points, int num_points) {
        mqi::vec3<float> pos0 = contour_points[0];
        mqi::vec3<float> pos1;
        float            min_y, max_y, max_x, intersect_x;
        int              count = 0;
        int              i, j, c = 0;
        for (i = 0, j = num_points - 1; i < num_points; j = i++) {
            pos0 = contour_points[i];
            pos1 = contour_points[j];
            if ((((pos0.y <= pos.y) && (pos.y < pos1.y)) ||
                 ((pos1.y <= pos.y) && (pos.y < pos0.y))) &&
                (pos.x < (pos1.x - pos0.x) * (pos.y - pos0.y) / (pos1.y - pos0.y) + pos0.x)) {
                c = !c;
            }
        }
        return c;
    }
    CUDA_HOST_DEVICE
    void
    fill_contour(uint8_t*           volume_contour,
                 mqi::vec3<float>*& contour_points,
                 int                num_contour_points,
                 mqi::vec3<ijk_t>   dim,
                 const R*           x_pix,
                 const R*           y_pix,
                 const R*           z_pix,
                 float              dx,
                 float              dy) {
        int x_ind = 0, y_ind = 0, z_ind = -1;
        for (int i = 0; i < dim.z - 1; i++) {
            if (contour_points[0].z > z_pix[i] && contour_points[0].z < z_pix[i + 1]) {
                z_ind = i;
                break;
            }
        }
        mqi::vec2<float> pos;
        uint32_t         idx = 0;
        float            x_pos, y_pos;
        if (z_ind >= 0) {
            for (x_ind = 0; x_ind < dim.x - 1; x_ind++) {
                for (y_ind = 0; y_ind < dim.y - 1; y_ind++) {
                    pos.x = x_pix[x_ind] + dx * 0.5;
                    pos.y = y_pix[y_ind] + dy * 0.5;
                    idx   = z_ind * dim.x * dim.y + y_ind * dim.x + x_ind;
                    assert(idx < dim.x * dim.y * dim.z);
                    if (sol1_1(pos, contour_points, num_contour_points)) {
                        volume_contour[idx] = 1;
                    } else {
                        //                        volume_contour[idx] = 0;
                    }
                }
            }
        }
    }

    CUDA_HOST
    double*
    reshape_data(int c_ind, int s_ind, mqi::vec3<ijk_t> dim) {
        //        R* reshaped_data = new R[dim.x * dim.y * dim.z];
        double* reshaped_data = new double[dim.x * dim.y * dim.z];
        //        std::memset(reshaped_data, 0, sizeof(R) * dim.x * dim.y * dim.z);
        int ind_x = 0, ind_y = 0, ind_z = 0, lin = 0;
        for (int i = 0; i < dim.x * dim.y * dim.z; i++) {
            reshaped_data[i] = 0;
        }
        printf("max capacity %d\n", this->world->children[c_ind]->scorers[s_ind]->max_capacity_);
        for (int ind = 0; ind < this->world->children[c_ind]->scorers[s_ind]->max_capacity_;
             ind++) {
            if (this->world->children[c_ind]->scorers[s_ind]->data_[ind].key1 != mqi::empty_pair &&
                this->world->children[c_ind]->scorers[s_ind]->data_[ind].key2 != mqi::empty_pair) {
                reshaped_data[this->world->children[c_ind]->scorers[s_ind]->data_[ind].key1] +=
                  this->world->children[c_ind]->scorers[s_ind]->data_[ind].value;
            }
        }
        return reshaped_data;
    }

    CUDA_HOST
    void
    save_reshaped_files() {
        auto                     start = std::chrono::high_resolution_clock::now();
        uint32_t                 vol_size;
        mqi::vec3<ijk_t>         dim;
        double*                  reshaped_data;
        std::string              filename;
        std::vector<std::string> beam_names = this->tx->get_beam_names();
        std::string              beam_name  = beam_names[bnb - 1];
        for (int c_ind = 0; c_ind < this->world->n_children; c_ind++) {
            for (int s_ind = 0; s_ind < this->world->children[c_ind]->n_scorers; s_ind++) {
                if (!this->world->children[c_ind]->scorers[s_ind]->save_output) {
                    continue;
                }
                filename = beam_name + "_" + std::to_string(c_ind) + "_" +
                           this->world->children[c_ind]->scorers[s_ind]->name_;
                dim           = this->world->children[c_ind]->geo->get_nxyz();
                vol_size      = dim.x * dim.y * dim.z;
                reshaped_data = this->reshape_data(c_ind, s_ind, dim);
                if (!this->output_format.compare("mhd")) {
                    mqi::io::save_to_mhd<R>(this->world->children[c_ind],
                                            reshaped_data,
                                            this->particles_per_history * this->rbe *
                                              this->n_fractions,
                                            this->output_path,
                                            filename,
                                            vol_size);
                } else if (!this->output_format.compare("mha")) {
                    mqi::io::save_to_mha<R>(this->world->children[c_ind],
                                            reshaped_data,
                                            this->particles_per_history * this->rbe *
                                              this->n_fractions,
                                            this->output_path,
                                            filename,
                                            vol_size);
                } else {
                    mqi::io::save_to_bin<double>(reshaped_data,
                                                 this->particles_per_history * this->rbe *
                                                   this->n_fractions,
                                                 this->output_path,
                                                 filename,
                                                 vol_size);
                }

                delete[] reshaped_data;
            }
        }
        auto                                      stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = stop - start;
        printf("Reshape and save done %f s\n", duration.count() * 0.0001);
    }

    CUDA_HOST
    virtual void
    save_sparse_file() {
        auto                     start = std::chrono::high_resolution_clock::now();
        mqi::vec3<ijk_t>         dim;
        std::string              filename;
        std::vector<std::string> beam_names = this->tx->get_beam_names();
        std::string              beam_name  = beam_names[bnb - 1];
        printf("%d\n", this->num_spots);
        for (int c_ind = 0; c_ind < this->world->n_children; c_ind++) {
            for (int s_ind = 0; s_ind < this->world->children[c_ind]->n_scorers; s_ind++) {
                filename = beam_name + "_" + std::to_string(c_ind) + "_" +
                           this->world->children[c_ind]->scorers[s_ind]->name_;
                dim = this->world->children[c_ind]->geo->get_nxyz();
                mqi::io::save_to_npz<R>(this->world->children[c_ind]->scorers[s_ind],
                                        this->particles_per_history * this->rbe * this->n_fractions,
                                        this->output_path,
                                        filename,
                                        dim,
                                        this->num_spots);
            }
        }
        auto                                      stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = stop - start;
        printf("Reshape and save to npz done %f s\n", duration.count() * 0.0001);
    }

    CUDA_HOST
    virtual void
    save_files() {
        ///< Get node pointer for phantom as we know it's id w.r.t world
        /// Binary output only
        auto                     start = std::chrono::high_resolution_clock::now();
        uint32_t                 vol_size;
        mqi::vec3<ijk_t>         dim;
        R*                       reshaped_data;
        std::string              filename;
        std::vector<std::string> beam_names = this->tx->get_beam_names();
        std::string              beam_name  = beam_names[bnb - 1];
        for (int c_ind = 0; c_ind < this->world->n_children; c_ind++) {
            for (int s_ind = 0; s_ind < this->world->children[c_ind]->n_scorers; s_ind++) {
                filename = beam_name + "_" + std::to_string(c_ind) + "_" +
                           this->world->children[c_ind]->scorers[s_ind]->name_;
                dim      = this->world->children[c_ind]->geo->get_nxyz();
                vol_size = dim.x * dim.y * dim.z;
                mqi::io::save_to_bin<R>(this->world->children[c_ind]->scorers[s_ind],
                                        this->particles_per_history * this->rbe * this->n_fractions,
                                        this->output_path,
                                        filename);
            }
        }
        auto                                      stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = stop - start;
        printf("Save done %f s\n", duration.count() * 0.0001);
    }
};

}   // namespace mqi

#endif
