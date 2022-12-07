#ifndef MQI_FILE_HANDLER_HPP
#define MQI_FILE_HANDLER_HPP

#include <fstream>
#include <iostream>
#include <moqui/base/mqi_beam_module_ion.hpp>
#include <moqui/base/mqi_roi.hpp>
#include <moqui/base/mqi_utils.hpp>

namespace mqi
{

class mask_reader
{
public:
    std::vector<std::string> mask_filenames;
    mqi::vec3<ijk_t>         ct_dim;
    size_t                   ct_size;
    uint8_t*                 mask_total;
    mask_reader() {
        ;
    }

    mask_reader(mqi::vec3<ijk_t> ct_dim) {
        this->ct_dim  = ct_dim;
        this->ct_size = ct_dim.x * ct_dim.y * ct_dim.z;
    }
    mask_reader(std::vector<std::string> filelist, mqi::vec3<ijk_t> ct_dim) {
        this->mask_filenames = filelist;
        this->ct_dim         = ct_dim;
        this->ct_size        = ct_dim.x * ct_dim.y * ct_dim.z;
    }

    ~mask_reader() {
        ;
    }

    CUDA_HOST
    uint8_t*
    read_mha_file(std::string filename) {
        std::string   line;
        std::ifstream fid(filename, std::ios::ate);
        fid.seekg(0);
        std::string delimeter = "=";
        size_t      pos, pos_current, pos_prev, total_size;
        std::string parameter, value;
        uint16_t    nx, ny, nz;
        uint8_t*    mask;
        while (std::getline(fid, line)) {
            line      = trim_copy(line);
            pos       = line.find(delimeter);
            parameter = trim_copy(line.substr(0, pos));
            if (strcasecmp(parameter.c_str(), "ObjectType") == 0) {
            } else if (strcasecmp(parameter.c_str(), "NDims") == 0) {
            } else if (strcasecmp(parameter.c_str(), "BinaryData") == 0) {
            } else if (strcasecmp(parameter.c_str(), "BinaryDataByteOrderMSB") == 0) {
            } else if (strcasecmp(parameter.c_str(), "CompressedData") == 0) {
            } else if (strcasecmp(parameter.c_str(), "TransformMatrix") == 0) {
            } else if (strcasecmp(parameter.c_str(), "Offset") == 0) {
            } else if (strcasecmp(parameter.c_str(), "CenterOfRotation") == 0) {
            } else if (strcasecmp(parameter.c_str(), "AnatomicalOrientation") == 0) {
            } else if (strcasecmp(parameter.c_str(), "ElementSpacing") == 0) {
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
        total_size = nx * ny * nz;
        mask       = new uint8_t[total_size];
        fid.read((char*) (&mask[0]), total_size * sizeof(mask[0]));
        fid.close();
        return mask;
    }

    CUDA_HOST
    CUDA_HOST
    void
    read_mask_files() {
        std::string filename;
        this->mask_total = new uint8_t[this->ct_size];
        for (int i = 0; i < this->ct_size; i++) {
            this->mask_total[i] = 0;
        }
        uint8_t* mask_temp;
        if (mask_filenames.empty()) {
            throw std::runtime_error("Mask filelist are required for masking scorers.");
        }
        for (int f = 0; f < mask_filenames.size(); f++) {
            filename = mask_filenames[f];
            printf("Reading maskfile %s\n", filename.c_str());
            mask_temp = read_mha_file(filename);
            for (int i = 0; i < this->ct_size; i++) {
                assert(mask_temp[i] >= 0 && mask_temp[i] <= 1);
                this->mask_total[i] += mask_temp[i];
            }
        }
    }

    CUDA_HOST
    int32_t*
    mask_mapping(uint8_t* mask) {
        int32_t* scorer_idx = new int32_t[this->ct_size];
        uint32_t count      = 0;
        for (uint32_t ind = 0; ind < this->ct_size; ind++) {
            if (mask[ind] > 0) {
                scorer_idx[ind] = count;
                count++;
            } else {
                scorer_idx[ind] = -1;
            }
        }
        return scorer_idx;
    }

    CUDA_HOST
    int32_t*
    mask_mapping() {
        int32_t* scorer_idx = new int32_t[this->ct_size];
        for (uint32_t ind = 0; ind < this->ct_size; ind++) {
            scorer_idx[ind] = ind;
        }
        return scorer_idx;
    }

    CUDA_HOST
    uint32_t
    size(int32_t* scorer_idx) {
        int count = 0;
        for (uint32_t ind = 0; ind < this->ct_size; ind++) {
            if (scorer_idx[ind] >= 0) { count++; }
        }
        return count;
    }

    CUDA_HOST
    void
    save_map(std::string filename, int32_t* scorer_idx) {
        std::ofstream fid0(filename);
        for (uint32_t ind = 0; ind < this->ct_size; ind++) {
            if (scorer_idx[ind] >= 0) { fid0 << ind << " " << scorer_idx[ind] << "\n"; }
        }
        fid0.close();
    }
    CUDA_HOST
    void
    set_mask(uint8_t* mask) {
        this->mask_total = mask;
    }

    CUDA_HOST
    cnb_t
    ijk2cnb(ijk_t i, ijk_t j, ijk_t k) {
        return k * this->ct_dim.x * this->ct_dim.y + j * this->ct_dim.x + i;
    }
    CUDA_HOST
    mqi::roi_t*
    mask_to_roi() {
        uint32_t              original_size = this->ct_size;
        uint32_t              length        = 0;
        mqi::roi_mapping_t    method        = mqi::CONTOUR;
        std::vector<uint32_t> start_vec;
        std::vector<uint32_t> stride_vec;
        std::vector<uint32_t> acc_stride_vec;
        bool                  ind_started = false;
        int32_t               cnb, start_ind, acc_stride_tmp;
        for (ijk_t z_ind = 0; z_ind < this->ct_dim.z; z_ind++) {
            for (ijk_t y_ind = 0; y_ind < this->ct_dim.y; y_ind++) {
                for (ijk_t x_ind = 0; x_ind < this->ct_dim.x; x_ind++) {
                    if (this->mask_total[ijk2cnb(x_ind, y_ind, z_ind)] == 1 && !ind_started) {
                        ind_started = true;
                        start_vec.push_back(ijk2cnb(x_ind, y_ind, z_ind));
                    }
                    if (this->mask_total[ijk2cnb(x_ind, y_ind, z_ind)] == 0 && ind_started) {
                        ind_started = false;
                        if (acc_stride_vec.size() > 0)
                            acc_stride_tmp = acc_stride_vec.back();
                        else
                            acc_stride_tmp = 0;
                        start_ind = start_vec.back();
                        stride_vec.push_back(ijk2cnb(x_ind, y_ind, z_ind) - start_ind);
                        acc_stride_vec.push_back(acc_stride_tmp + stride_vec.back());
                    }
                }
            }
        }
        uint32_t* start      = new uint32_t[start_vec.size()];
        uint32_t* stride     = new uint32_t[start_vec.size()];
        uint32_t* acc_stride = new uint32_t[acc_stride_vec.size()];

        std::copy(start_vec.begin(), start_vec.end(), start);
        std::copy(stride_vec.begin(), stride_vec.end(), stride);
        std::copy(acc_stride_vec.begin(), acc_stride_vec.end(), acc_stride);
        length = start_vec.size();
        //        mqi::roi_t roi(method, original_size, length, start, stride);
        mqi::roi_t* roi = new mqi::roi_t(method, original_size, length, start, stride, acc_stride);
        return roi;
    }
};

class file_parser
{
public:
    std::string              filename;
    std::string              delimeter;
    std::vector<std::string> parameters_total;

    file_parser(std::string filename, std::string delimeter) {
        this->filename         = filename;
        this->delimeter        = delimeter;
        this->parameters_total = read_input_parameters();
    }
    ~file_parser() {
        ;
    }

    CUDA_HOST
    std::vector<std::string>
    read_input_parameters() {
        std::ifstream            fid(this->filename);
        std::string              line;
        std::string              option, value;
        std::vector<std::string> parameters_total;
        size_t                   pos, comment;
        if (fid.is_open()) {
            while (getline(fid, line)) {
                if (line.size() == 0) continue;
                if(line.find_first_not_of(" \f\n\r\t\v\0\\")==std::string::npos) continue;
                line = trim_copy(line);
                if (line.at(0) == '#') continue;
                comment = line.find("#");
                if (comment != std::string::npos) { line = line.substr(0, comment); }
                parameters_total.push_back(line);
            }
            fid.close();
        } else {
            throw std::runtime_error("Cannot open input parameter file.");
        }
        return parameters_total;
    }

    CUDA_HOST
    std::string
    get_path(std::string filepath) {
        std::string delimeter   = "/";
        std::string path        = "";
        size_t      pos_current = 0, pos_prev = 0;
        pos_current = filepath.find(delimeter);
        while (pos_current != std::string::npos) {
            if (path.size() == 0) {
                path += filepath.substr(pos_prev, pos_current - pos_prev);
            } else {
                path += "/" + filepath.substr(pos_prev, pos_current - pos_prev);
            }

            pos_prev    = pos_current + 1;
            pos_current = filepath.find(delimeter, pos_prev);
        }
        return path;
    }

    CUDA_HOST
    std::string
    get_string(std::string option, std::string default_value) {
        size_t      pos;
        std::string option_in, value = default_value;
        for (int line = 0; line < parameters_total.size(); line++) {
            pos = parameters_total[line].find(delimeter);
            if (pos != std::string::npos) {
                option_in = trim_copy(parameters_total[line].substr(0, pos));
                if (strcasecmp(option.c_str(), option_in.c_str()) == 0 && (pos+1) != parameters_total[line].length() && parameters_total[line].substr(pos + 1, std::string::npos).find_first_not_of(" \f\n\r\t\v\0\\")!=std::string::npos) {
                    value = trim_copy(parameters_total[line].substr(pos + 1, std::string::npos));
                    return value;
                }
            }
        }
        return value;
    }

    CUDA_HOST
    std::vector<std::string>
    get_string_vector(std::string option, std::string delimeter1) {
        size_t                   pos_current = 0, pos_prev = 0;
        std::string              value = get_string(option, "");
        std::vector<std::string> values;
        pos_current = value.find(delimeter1);
        std::string temp;
        while (pos_current != std::string::npos) {
            temp = trim_copy(value.substr(pos_prev, pos_current - pos_prev));
            if (temp.size() > 0) { values.push_back(temp); }
            pos_prev    = pos_current + 1;
            pos_current = value.find(delimeter1, pos_prev);
        }
        temp = trim_copy(value.substr(pos_prev, pos_current - pos_prev));
        if (temp.size() > 0) { values.push_back(temp); }
        return values;
    }

    CUDA_HOST
    float
    get_float(std::string option, float default_value) {
        float value = std::atof(get_string(option, std::to_string(default_value)).c_str());
        return value;
    }

    CUDA_HOST
    std::vector<float>
    get_float_vector(std::string option, std::string delimter) {
        std::vector<std::string> value_tmp = get_string_vector(option, delimeter);
        if (value_tmp.size() > 0) {
            std::vector<float> value;
            for (int i = 0; i < value_tmp.size(); i++) {
                value.push_back(std::atof(value_tmp[i].c_str()));
            }
            return value;
        } else {
            std::vector<float> empty(0);
            return empty;
        }
    }

    CUDA_HOST
    int
    get_int(std::string option, int default_value) {
        int value = std::atoi(get_string(option, std::to_string(default_value)).c_str());
        return value;
    }

    CUDA_HOST
    std::vector<int>
    get_int_vector(std::string option, std::string delimeter) {
        std::vector<std::string> value_tmp = get_string_vector(option, delimeter);
        if (value_tmp.size() > 0) {
            std::vector<int> value;
            for (int i = 0; i < value_tmp.size(); i++) {
                value.push_back(std::atoi(value_tmp[i].c_str()));
            }
            return value;
        } else {
            std::vector<int> empty(0);
            return empty;
        }
    }

    CUDA_HOST
    bool
    get_bool(std::string option, bool default_value) {
        std::string temp  = get_string(option, std::to_string(default_value));
        bool        value = strcasecmp(temp.c_str(), "true") == 0 || std::atoi(temp.c_str()) != 0;
        return value;
    }

    CUDA_HOST
    scorer_t
    string_to_scorer_type(std::string scorer_name) {
        scorer_t type = mqi::VIRTUAL;
        if (strcasecmp(scorer_name.c_str(), "EnergyDeposition") == 0) {
            type = mqi::ENERGY_DEPOSITION;
        } else if (strcasecmp(scorer_name.c_str(), "Dose") == 0) {
            type = mqi::DOSE;
        } else if (strcasecmp(scorer_name.c_str(), "LETd") == 0) {
            type = mqi::LETd;
        } else if (strcasecmp(scorer_name.c_str(), "LETt") == 0) {
            type = mqi::LETt;
        } else if (strcasecmp(scorer_name.c_str(), "Dij") == 0) {
            type = mqi::DOSE_Dij;
        } else if (strcasecmp(scorer_name.c_str(), "TrackLength") == 0) {
            type = mqi::TRACK_LENGTH;
        } else {
            throw std::runtime_error("Unrecognized scorer name");
        }
        return type;
    }

    CUDA_HOST
    aperture_type_t
    string_to_aperture_type(std::string aperture_string) {
        aperture_type_t type;
        if (strcasecmp(aperture_string.c_str(), "VOLUME") == 0) {
            type = mqi::VOLUME;
        } else if (strcasecmp(aperture_string.c_str(), "MASK") == 0) {
            type = mqi::MASK;
        } else {
            throw std::runtime_error("Undefined aperture type.");
        }
        return type;
    }

    CUDA_HOST
    sim_type_t
    string_to_sim_type(std::string sim_name) {
        sim_type_t type;
        if (strcasecmp(sim_name.c_str(), "perBeam") == 0) {
            type = mqi::PER_BEAM;
        } else if (strcasecmp(sim_name.c_str(), "perSpot") == 0) {
            type = mqi::PER_SPOT;
        }
        return type;
    }
};
}   // namespace mqi

#endif
