#ifndef MQI_IO_HPP
#define MQI_IO_HPP

#include <algorithm>
#include <complex>
#include <cstdint>
#include <iomanip>   // std::setprecision
#include <iostream>
#include <numeric>   //accumulate
#include <valarray>
#include <zlib.h>

#include <sys/mman.h>   //for io

#include <moqui/base/mqi_common.hpp>
#include <moqui/base/mqi_hash_table.hpp>
#include <moqui/base/mqi_roi.hpp>
#include <moqui/base/mqi_sparse_io.hpp>
#include <moqui/base/mqi_scorer.hpp>

namespace mqi
{
namespace io
{
///<  save scorer data to a file in binary format
///<  scr: scorer pointer
///<  scale: data will be multiplied by
///<  dir  : directory path. file name will be dir + scr->name + ".bin"
///<  reshape: roi is used in scorer, original size will be defined.
template<typename R>
void
save_to_bin(const mqi::scorer<R>* src,
            const R               scale,
            const std::string&    filepath,
            const std::string&    filename);

template<typename R>
void
save_to_bin(const R*           src,
            const R            scale,
            const std::string& filepath,
            const std::string& filename,
            const uint32_t     length);

template<typename R>
void
save_to_npz(const mqi::scorer<R>* src,
            const R               scale,
            const std::string&    filepath,
            const std::string&    filename,
            mqi::vec3<mqi::ijk_t> dim,
            uint32_t              num_spots);

template<typename R>
void
save_to_npz2(const mqi::scorer<R>* src,
             const R               scale,
             const std::string&    filepath,
             const std::string&    filename,
             mqi::vec3<mqi::ijk_t> dim,
             uint32_t              num_spots);

template<typename R>
void
save_to_npz(const mqi::scorer<R>* src,
            const R               scale,
            const std::string&    filepath,
            const std::string&    filename,
            mqi::vec3<mqi::ijk_t> dim,
            uint32_t              num_spots,
            R*                    time_scale,
            R                     threshold);

template<typename R>
void
save_to_bin(const mqi::key_value* src,
            const R               scale,
            uint32_t              max_capacity,
            const std::string&    filepath,
            const std::string&    filename);

template<typename R>
void
save_to_mhd(const mqi::node_t<R>* children,
            const double*         src,
            const R               scale,
            const std::string&    filepath,
            const std::string&    filename,
            const uint32_t        length);

template<typename R>
void
save_to_mha(const mqi::node_t<R>* children,
            const double*         src,
            const R               scale,
            const std::string&    filepath,
            const std::string&    filename,
            const uint32_t        length);
}   // namespace io
}   // namespace mqi

///< Function to write key values into file
///< src: array and this array is copied
///<
template<typename R>
void
mqi::io::save_to_bin(const mqi::scorer<R>* src,
                     const R               scale,
                     const std::string&    filepath,
                     const std::string&    filename) {
    /// create a copy using valarray and apply scale

    unsigned int            nnz = 0;
    std::vector<mqi::key_t> key1;
    std::vector<mqi::key_t> key2;
    std::vector<double>     value;
    key1.clear();
    key2.clear();
    value.clear();
    for (int ind = 0; ind < src->max_capacity_; ind++) {
        if (src->data_[ind].key1 != mqi::empty_pair && src->data_[ind].key2 != mqi::empty_pair &&
            src->data_[ind].value > 0) {
            key1.push_back(src->data_[ind].key1);
            key2.push_back(src->data_[ind].key2);
            value.push_back(src->data_[ind].value * scale);
        }
    }

    printf("length %lu %lu %lu\n", key1.size(), key2.size(), value.size());

    /// open out stream
    std::ofstream fid_key1(filepath + "/" + filename + "_key1.raw",
                           std::ios::out | std::ios::binary);
    if (!fid_key1)
        std::cout << "Cannot write :" << filepath + "/" + filename + "_key1.raw" << std::endl;

    /// write to a file
    fid_key1.write(reinterpret_cast<const char*>(&key1.data()[0]),
                   key1.size() * sizeof(mqi::key_t));
    fid_key1.close();

    std::ofstream fid_key2(filepath + "/" + filename + "_key2.raw",
                           std::ios::out | std::ios::binary);
    if (!fid_key2)
        std::cout << "Cannot write :" << filepath + "/" + filename + "_key2.raw" << std::endl;

    /// write to a file
    fid_key2.write(reinterpret_cast<const char*>(&key2.data()[0]),
                   key2.size() * sizeof(mqi::key_t));
    fid_key2.close();

    std::ofstream fid_bin(filepath + "/" + filename + "_value.raw",
                          std::ios::out | std::ios::binary);
    if (!fid_bin)
        std::cout << "Cannot write :" << filepath + "/" + filename + "_value.raw" << std::endl;

    /// write to a file
    fid_bin.write(reinterpret_cast<const char*>(&value.data()[0]), value.size() * sizeof(double));
    fid_bin.close();
}

///< Function to write array into file
///< src: array and this array is copied
///<
template<typename R>
void
mqi::io::save_to_bin(const R*           src,
                     const R            scale,
                     const std::string& filepath,
                     const std::string& filename,
                     const uint32_t     length) {
    /// create a copy using valarray and apply scale
    std::valarray<R> dest(src, length);
    munmap(&dest, length * sizeof(R));
    dest *= scale;
    /// open out stream
    std::ofstream fid_bin(filepath + "/" + filename + ".raw", std::ios::out | std::ios::binary);
    if (!fid_bin) std::cout << "Cannot write :" << filepath + "/" + filename + ".raw" << std::endl;

    /// write to a file
    fid_bin.write(reinterpret_cast<const char*>(&dest[0]), length * sizeof(R));
    fid_bin.close();
}

///< Function to write key values into file
///< src: array and this array is copied
///<
template<typename R>
void
mqi::io::save_to_bin(const mqi::key_value* src,
                     const R               scale,
                     uint32_t              max_capacity,
                     const std::string&    filepath,
                     const std::string&    filename) {
    /// create a copy using valarray and apply scale

    unsigned int            nnz = 0;
    std::vector<mqi::key_t> key1;
    std::vector<mqi::key_t> key2;
    std::vector<R>          value;
    key1.clear();
    key2.clear();
    value.clear();
    for (int ind = 0; ind < max_capacity; ind++) {
        if (src[ind].key1 != mqi::empty_pair && src[ind].key2 != mqi::empty_pair &&
            src[ind].value > 0) {
            key1.push_back(src[ind].key1);
            key2.push_back(src[ind].key2);
            value.push_back(src[ind].value * scale);
        }
    }

    printf("length %lu %lu %lu\n", key1.size(), key2.size(), value.size());
    /// open out stream
    std::ofstream fid_key1(filepath + "/" + filename + "_key1.raw",
                           std::ios::out | std::ios::binary);
    if (!fid_key1)
        std::cout << "Cannot write :" << filepath + "/" + filename + "_key1.raw" << std::endl;

    /// write to a file
    fid_key1.write(reinterpret_cast<const char*>(&key1.data()[0]),
                   key1.size() * sizeof(mqi::key_t));
    fid_key1.close();

    std::ofstream fid_key2(filepath + "/" + filename + "_key2.raw",
                           std::ios::out | std::ios::binary);
    if (!fid_key2)
        std::cout << "Cannot write :" << filepath + "/" + filename + "_key2.raw" << std::endl;

    /// write to a file
    fid_key2.write(reinterpret_cast<const char*>(&key2.data()[0]),
                   key2.size() * sizeof(mqi::key_t));
    fid_key2.close();

    std::ofstream fid_bin(filepath + "/" + filename + "_value.raw",
                          std::ios::out | std::ios::binary);
    if (!fid_bin)
        std::cout << "Cannot write :" << filepath + "/" + filename + "_value.raw" << std::endl;

    /// write to a file
    fid_bin.write(reinterpret_cast<const char*>(&value.data()[0]), value.size() * sizeof(R));
    fid_bin.close();
}

///< Function to write key values into file
///< src: array and this array is copied
///<

template<typename R>
void
mqi::io::save_to_npz(const mqi::scorer<R>* src,
                     const R               scale,
                     const std::string&    filepath,
                     const std::string&    filename,
                     mqi::vec3<mqi::ijk_t> dim,
                     uint32_t              num_spots) {
    uint32_t vol_size;
    vol_size = dim.x * dim.y * dim.z;

    /// create a copy using valarray and apply scale
    const std::string name_a = "indices.npy", name_b = "indptr.npy", name_c = "shape.npy",
                      name_d = "data.npy", name_e = "format.npy";
    std::vector<double>* value_vec = new std::vector<double>[num_spots];
    std::vector<mqi::key_t>*          vox_vec = new std::vector<mqi::key_t>[num_spots];
    std::vector<double>               data_vec;
    std::vector<uint32_t>             indices_vec;
    std::vector<uint32_t>             indptr_vec;
    mqi::key_t                        vox_ind, spot_ind;
    double                            value;
    int                               spot_start = 0, spot_end = 0;
    int                               vox_in_spot[num_spots];
    std::vector<double>::iterator     it_data;
    std::vector<uint32_t>::iterator   it_ind;
    std::vector<mqi::key_t>::iterator it_spot;
    int                               vox_count;
    printf("save_to_npz\n");

    printf("scan start %d\n", src->max_capacity_);
    for (int ind = 0; ind < src->max_capacity_; ind++) {
        if (src->data_[ind].key1 != mqi::empty_pair && src->data_[ind].key2 != mqi::empty_pair) {
            vox_count = 0;
            vox_ind   = src->data_[ind].key1;
            spot_ind  = src->data_[ind].key2;
            assert(vox_ind >= 0 && vox_ind < vol_size);
            value = src->data_[ind].value;
            value_vec[spot_ind].push_back(value * scale);
            vox_vec[spot_ind].push_back(vox_ind);
        }
    }

    vox_count = 0;
    indptr_vec.push_back(vox_count);
    for (int ii = 0; ii < num_spots; ii++) {
        data_vec.insert(data_vec.end(), value_vec[ii].begin(), value_vec[ii].end());
        indices_vec.insert(indices_vec.end(), vox_vec[ii].begin(), vox_vec[ii].end());
        vox_count += vox_vec[ii].size();
        indptr_vec.push_back(vox_count);
    }

    printf("scan done %lu %lu %lu\n", data_vec.size(), indices_vec.size(), indptr_vec.size());
    printf("%d %d\n", vol_size, num_spots);

    uint32_t    shape[2] = { num_spots, vol_size };
    std::string format   = "csr";
    size_t      size_a = indices_vec.size(), size_b = indptr_vec.size(), size_c = 2,
           size_d = data_vec.size(), size_e = 3;

    uint32_t* indices = new uint32_t[indices_vec.size()];
    uint32_t* indptr  = new uint32_t[indptr_vec.size()];
    double*   data    = new double[data_vec.size()];
    std::copy(indices_vec.begin(), indices_vec.end(), indices);
    std::copy(indptr_vec.begin(), indptr_vec.end(), indptr);
    std::copy(data_vec.begin(), data_vec.end(), data);
    printf("%lu\n", size_b);
    mqi::io::save_npz(filepath + "/" + filename + ".npz", name_a, indices, size_a, "w");
    mqi::io::save_npz(filepath + "/" + filename + ".npz", name_b, indptr, size_b, "a");
    mqi::io::save_npz(filepath + "/" + filename + ".npz", name_c, shape, size_c, "a");
    mqi::io::save_npz(filepath + "/" + filename + ".npz", name_d, data, size_d, "a");
    mqi::io::save_npz(filepath + "/" + filename + ".npz", name_e, format, size_e, "a");
}

template<typename R>
void
mqi::io::save_to_npz2(const mqi::scorer<R>* src,
                      const R               scale,
                      const std::string&    filepath,
                      const std::string&    filename,
                      mqi::vec3<mqi::ijk_t> dim,
                      uint32_t              num_spots) {
    uint32_t vol_size;
    vol_size = src->roi_->get_mask_size();
    /// create a copy using valarray and apply scale
    const std::string name_a = "indices.npy", name_b = "indptr.npy", name_c = "shape.npy",
                      name_d = "data.npy", name_e = "format.npy";

    std::vector<double>*              value_vec = new std::vector<double>[vol_size];
    std::vector<mqi::key_t>*          spot_vec  = new std::vector<mqi::key_t>[vol_size];
    std::vector<double>               data_vec;
    std::vector<uint32_t>             indices_vec;
    std::vector<uint32_t>             indptr_vec;
    mqi::key_t                        vox_ind, spot_ind;
    double                            value;
    int                               spot_start = 0, spot_end = 0;
    std::vector<double>::iterator     it_data;
    std::vector<uint32_t>::iterator   it_ind;
    std::vector<mqi::key_t>::iterator it_spot;
    int                               spot_count;
    printf("save_to_npz\n");

    printf("scan start %d\n", src->max_capacity_);
    for (int ind = 0; ind < src->max_capacity_; ind++) {
        if (src->data_[ind].key1 != mqi::empty_pair && src->data_[ind].key2 != mqi::empty_pair) {
            vox_ind = src->data_[ind].key1;
            vox_ind = src->roi_->get_mask_idx(vox_ind);
            if (vox_ind < 0) {
                printf("is this right?\n");
                continue;
            }
            spot_ind = src->data_[ind].key2;
            assert(vox_ind >= 0 && vox_ind < vol_size);
            value = src->data_[ind].value;
            assert(value > 0);
            value_vec[vox_ind].push_back(value * scale);
            spot_vec[vox_ind].push_back(spot_ind);
        }
    }
    printf("Sorting start\n");
    for (int ind = 0; ind < vol_size; ind++) {
        if (spot_vec[ind].size() > 1) {
            std::vector<int> sort_ind(spot_vec[ind].size());
            std::iota(sort_ind.begin(), sort_ind.end(), 0);
            sort(sort_ind.begin(), sort_ind.end(), [&](int i, int j) {
                return spot_vec[ind][i] < spot_vec[ind][j];
            });
            std::vector<double>     sorted_value(spot_vec[ind].size());
            std::vector<mqi::key_t> sorted_spot(spot_vec[ind].size());
            for (int sorted_ind = 0; sorted_ind < spot_vec[ind].size(); sorted_ind++) {
                sorted_value[sorted_ind] = value_vec[ind][sort_ind[sorted_ind]];
                sorted_spot[sorted_ind]  = spot_vec[ind][sort_ind[sorted_ind]];
            }
            spot_vec[ind]  = sorted_spot;
            value_vec[ind] = sorted_value;
        }
    }

    spot_count = 0;
    indptr_vec.push_back(spot_count);
    for (int ii = 0; ii < vol_size; ii++) {
        data_vec.insert(data_vec.end(), value_vec[ii].begin(), value_vec[ii].end());
        indices_vec.insert(indices_vec.end(), spot_vec[ii].begin(), spot_vec[ii].end());
        spot_count += spot_vec[ii].size();
        indptr_vec.push_back(spot_count);
    }

    uint32_t    shape[2] = { vol_size, num_spots };
    std::string format   = "csr";
    size_t      size_a = indices_vec.size(), size_b = indptr_vec.size(), size_c = 2,
           size_d = data_vec.size(), size_e = 3;

    uint32_t* indices = new uint32_t[indices_vec.size()];
    uint32_t* indptr  = new uint32_t[indptr_vec.size()];
    double*   data    = new double[data_vec.size()];
    std::copy(indices_vec.begin(), indices_vec.end(), indices);
    std::copy(indptr_vec.begin(), indptr_vec.end(), indptr);
    std::copy(data_vec.begin(), data_vec.end(), data);
    printf("%lu\n", size_b);
    mqi::io::save_npz(filepath + "/" + filename + ".npz", name_a, indices, size_a, "w");
    mqi::io::save_npz(filepath + "/" + filename + ".npz", name_b, indptr, size_b, "a");
    mqi::io::save_npz(filepath + "/" + filename + ".npz", name_c, shape, size_c, "a");
    mqi::io::save_npz(filepath + "/" + filename + ".npz", name_d, data, size_d, "a");
    mqi::io::save_npz(filepath + "/" + filename + ".npz", name_e, format, size_e, "a");
}

template<typename R>
void
mqi::io::save_to_npz(const mqi::scorer<R>* src,
                     const R               scale,
                     const std::string&    filepath,
                     const std::string&    filename,
                     mqi::vec3<mqi::ijk_t> dim,
                     uint32_t              num_spots,
                     R*                    time_scale,
                     R                     threshold) {
    uint32_t vol_size;
    vol_size = dim.x * dim.y * dim.z;
    /// create a copy using valarray and apply scale
    const std::string name_a = "indices.npy", name_b = "indptr.npy", name_c = "shape.npy",
                      name_d = "data.npy", name_e = "format.npy";
    std::vector<double>               value_vec[num_spots];
    std::vector<mqi::key_t>           vox_vec[num_spots];
    std::vector<double>               data_vec;
    std::vector<uint32_t>             indices_vec;
    std::vector<uint32_t>             indptr_vec;
    mqi::key_t                        vox_ind, spot_ind;
    double                            value;
    int                               spot_start = 0, spot_end = 0;
    int                               vox_in_spot[num_spots];
    std::vector<double>::iterator     it_data;
    std::vector<uint32_t>::iterator   it_ind;
    std::vector<mqi::key_t>::iterator it_spot;
    int                               vox_count;
    printf("save_to_npz\n");
    for (int ind = 0; ind < num_spots; ind++) {
        vox_in_spot[ind] = 0;
    }
    printf("scan start %d\n", src->max_capacity_);
    for (int ind = 0; ind < src->max_capacity_; ind++) {
        if (src->data_[ind].key1 != mqi::empty_pair && src->data_[ind].key2 != mqi::empty_pair) {
            vox_count = 0;
            vox_ind   = src->data_[ind].key1;
            spot_ind  = src->data_[ind].key2;
            assert(vox_ind >= 0 && vox_ind < vol_size);
            value = src->data_[ind].value;
            value *= scale;
            value -= 2 * threshold;
            if (value < 0) value = 0;
            value /= time_scale[spot_ind];
            value_vec[spot_ind].push_back(value);
            vox_vec[spot_ind].push_back(vox_ind);
        }
    }

    vox_count = 0;
    indptr_vec.push_back(vox_count);
    for (int ii = 0; ii < num_spots; ii++) {
        data_vec.insert(data_vec.end(), value_vec[ii].begin(), value_vec[ii].end());
        indices_vec.insert(indices_vec.end(), vox_vec[ii].begin(), vox_vec[ii].end());
        vox_count += vox_vec[ii].size();
        indptr_vec.push_back(vox_count);
    }
    printf("scan done %lu %lu %lu\n", data_vec.size(), indices_vec.size(), indptr_vec.size());
    printf("%d %d\n", vol_size, num_spots);

    uint32_t    shape[2] = { num_spots, vol_size };
    std::string format   = "csr";
    size_t      size_a = indices_vec.size(), size_b = indptr_vec.size(), size_c = 2,
           size_d = data_vec.size(), size_e = 3;

    uint32_t* indices = new uint32_t[indices_vec.size()];
    uint32_t* indptr  = new uint32_t[indptr_vec.size()];
    double*   data    = new double[data_vec.size()];
    std::copy(indices_vec.begin(), indices_vec.end(), indices);
    std::copy(indptr_vec.begin(), indptr_vec.end(), indptr);
    std::copy(data_vec.begin(), data_vec.end(), data);
    printf("%lu\n", size_b);
    mqi::io::save_npz(filepath + "/" + filename + ".npz", name_a, indices, size_a, "w");
    mqi::io::save_npz(filepath + "/" + filename + ".npz", name_b, indptr, size_b, "a");
    mqi::io::save_npz(filepath + "/" + filename + ".npz", name_c, shape, size_c, "a");
    mqi::io::save_npz(filepath + "/" + filename + ".npz", name_d, data, size_d, "a");
    mqi::io::save_npz(filepath + "/" + filename + ".npz", name_e, format, size_e, "a");
}

template<typename R>
void
mqi::io::save_to_mhd(const mqi::node_t<R>* children,
                     const double*         src,
                     const R               scale,
                     const std::string&    filepath,
                     const std::string&    filename,
                     const uint32_t        length) {
    ///< TODO: this works only for two depth world
    ///< TODO: dx, dy, and dz calculation works only for AABB
    float dx = children->geo[0].get_x_edges()[1];
    dx -= children->geo[0].get_x_edges()[0];
    float dy = children->geo[0].get_y_edges()[1];
    dy -= children->geo[0].get_y_edges()[0];
    float dz = children->geo[0].get_z_edges()[1];
    dz -= children->geo[0].get_z_edges()[0];
    float x0 = children->geo[0].get_x_edges()[0];
    x0 += children->geo[0].get_x_edges()[0];
    x0 /= 2.0;
    float y0 = children->geo[0].get_y_edges()[0];
    y0 += children->geo[0].get_y_edges()[0];
    y0 /= 2.0;
    float z0 = children->geo[0].get_z_edges()[0];
    z0 += children->geo[0].get_z_edges()[0];
    z0 /= 2.0;
    std::ofstream fid_header(filepath + "/" + filename + ".mhd", std::ios::out);
    if (!fid_header) { std::cout << "Cannot open file!" << std::endl; }
    fid_header << "ObjectType = Image\n";
    fid_header << "NDims = 3\n";
    fid_header << "BinaryData = True\n";
    fid_header
      << "BinaryDataByteOrderMSB = False\n";   // True for big endian, False for little endian
    fid_header << "CompressedData = False\n";
    fid_header << "TransformMatrix 1 0 0 0 1 0 0 0 1\n";
    fid_header << "Offset " << x0 << " " << y0 << " " << z0 << std::endl;
    fid_header << "CenterOfRotation 0 0 0\n";
    fid_header << "AnatomicOrientation = RAI\n";
    fid_header << "DimSize = " << children->geo[0].get_nxyz().x << " "
               << children->geo[0].get_nxyz().y << " " << children->geo[0].get_nxyz().z << "\n";
    ///< TODO: if R is double, MET_FLOAT should be MET_DOUBLE
    fid_header << "ElementType = MET_DOUBLE\n";

    fid_header << "ElementSpacing = " << dx << " " << dy << " " << dz << "\n";
    fid_header << "ElementDataFile = " << filename << ".raw"
               << "\n";
    fid_header.close();
    if (!fid_header.good()) { std::cout << "Error occurred at writing time!" << std::endl; }
    std::valarray<double> dest(src, length);
    munmap(&dest, length * sizeof(double));
    dest *= scale;
    std::ofstream fid_raw(filepath + "/" + filename + ".raw", std::ios::out | std::ios::binary);
    if (!fid_raw) { std::cout << "Cannot open file!" << std::endl; }
    fid_raw.write(reinterpret_cast<const char*>(&dest[0]), length * sizeof(double));

    fid_raw.close();
    if (!fid_raw.good()) { std::cout << "Error occurred at writing time!" << std::endl; }
}

template<typename R>
void
mqi::io::save_to_mha(const mqi::node_t<R>* children,
                     const double*         src,
                     const R               scale,
                     const std::string&    filepath,
                     const std::string&    filename,
                     const uint32_t        length) {
    ///< TODO: this works only for two depth world
    ///< TODO: dx, dy, and dz calculation works only for AABB
    float dx = children->geo[0].get_x_edges()[1];
    dx -= children->geo[0].get_x_edges()[0];
    float dy = children->geo[0].get_y_edges()[1];
    dy -= children->geo[0].get_y_edges()[0];
    float dz = children->geo[0].get_z_edges()[1];
    dz -= children->geo[0].get_z_edges()[0];
    float x0 = children->geo[0].get_x_edges()[0] + dx * 0.5;
    float y0 = children->geo[0].get_y_edges()[0] + dy * 0.5;
    float z0 = children->geo[0].get_z_edges()[0] + dz * 0.5;
    std::cout << "x0 " << std::setprecision(9) << x0 << " y0 " << y0 << " z0 " << z0 << std::endl;
    std::valarray<double> dest(src, length);
    munmap(&dest, length * sizeof(double));
    dest *= scale;
    std::ofstream fid_header(filepath + "/" + filename + ".mha", std::ios::out);
    if (!fid_header) { std::cout << "Cannot open file!" << std::endl; }
    fid_header << "ObjectType = Image\n";
    fid_header << "NDims = 3\n";
    fid_header << "BinaryData = True\n";
    fid_header
      << "BinaryDataByteOrderMSB = False\n";   // True for big endian, False for little endian
    fid_header << "CompressedData = False\n";
    fid_header << "TransformMatrix = 1 0 0 0 1 0 0 0 1\n";
    fid_header << "Origin = " << std::setprecision(9) << x0 << " " << y0 << " " << z0 << "\n";
    fid_header << "CenterOfRotation = 0 0 0\n";
    fid_header << "AnatomicOrientation = RAI\n";
    fid_header << "DimSize = " << children->geo[0].get_nxyz().x << " "
               << children->geo[0].get_nxyz().y << " " << children->geo[0].get_nxyz().z << "\n";
    ///< TODO: if R is double, MET_FLOAT should be MET_DOUBLE
    fid_header << "ElementType = MET_DOUBLE\n";
    fid_header << "HeaderSize = -1\n";
    fid_header << "ElementSpacing = " << std::setprecision(9) << dx << " " << dy << " " << dz
               << "\n";
    fid_header << "ElementDataFile = LOCAL\n";
    fid_header.write(reinterpret_cast<const char*>(&dest[0]), length * sizeof(double));
    fid_header.close();
    if (!fid_header.good()) { std::cout << "Error occurred at writing time!" << std::endl; }
}

#endif
