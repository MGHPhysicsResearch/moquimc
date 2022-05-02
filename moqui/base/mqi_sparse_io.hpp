#ifndef MQI_SPARSE_IO_HPP
#define MQI_SPARSE_IO_HPP

#include <algorithm>
#include <complex>
#include <cstdint>
#include <iostream>
#include <numeric>   //accumulate
#include <valarray>
#include <zlib.h>

#include <sys/mman.h>   //for io

#include <moqui/base/mqi_common.hpp>
#include <moqui/base/mqi_hash_table.hpp>
#include <moqui/base/mqi_roi.hpp>
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
void
save_npz(std::string filename,
         std::string var_name,
         std::string data,
         size_t      shape,
         std::string mode);

template<typename T>
void
save_npz(std::string filename, std::string var_name, T* data, size_t shape, std::string mode);

void
push_value(std::vector<char>& vec, const std::string str);

void
push_value(std::vector<char>& vec, const uint16_t str);

void
push_value(std::vector<char>& vec, const uint32_t str);

void
parse_zip_footer(std::string filename,
                 uint16_t&   nrecs,
                 size_t&     global_header_size,
                 size_t&     global_header_offset);

char
map_type(const std::type_info& t);
}   // namespace io
}   // namespace mqi

///< Function to write array into file
///< src: array and this array is copied
///<

///< Function to write key values into file
///< src: array and this array is copied
///<

void
mqi::io::push_value(std::vector<char>& vec, const std::string str) {
    for (int i = 0; i < str.length(); i++) {
        vec.push_back(str[i]);
    }
}

void
mqi::io::push_value(std::vector<char>& vec, const uint16_t str) {
    for (size_t byte = 0; byte < sizeof(str); byte++) {
        char val = *((char*) &str + byte);
        vec.push_back(val);
    }
}

void
mqi::io::push_value(std::vector<char>& vec, const uint32_t str) {
    for (size_t byte = 0; byte < sizeof(str); byte++) {
        char val = *((char*) &str + byte);
        vec.push_back(val);
    }
}

void
mqi::io::parse_zip_footer(std::string filename,
                          uint16_t&   nrecs,
                          size_t&     global_header_size,
                          size_t&     global_header_offset) {
    std::ifstream     fp(filename.c_str(), std::ios::in | std::ios::binary);
    std::vector<char> footer(22);
    fp.seekg(-22, fp.end);
    fp.read(&footer[0], 22);
    if (!fp) {
        std::cout << "error: only " << fp.gcount() << " could be read" << std::endl;
        exit(-1);
    }
    uint16_t disk_no, disk_start, nrecs_on_disk, comment_len;
    disk_no              = *(uint16_t*) &footer[4];
    disk_start           = *(uint16_t*) &footer[6];
    nrecs_on_disk        = *(uint16_t*) &footer[8];
    nrecs                = *(uint16_t*) &footer[10];
    global_header_size   = *(uint32_t*) &footer[12];
    global_header_offset = *(uint32_t*) &footer[16];
    comment_len          = *(uint16_t*) &footer[20];
    assert(disk_no == 0);
    assert(disk_start == 0);
    assert(nrecs_on_disk == nrecs);
    assert(comment_len == 0);
    fp.seekg(global_header_offset, fp.beg);

    fp.close();
}

char
mqi::io::map_type(const std::type_info& t) {
    if (t == typeid(float))
        return 'f';
    else if (t == typeid(double))
        return 'f';
    else if (t == typeid(long double))
        return 'f';

    else if (t == typeid(int))
        return 'i';
    else if (t == typeid(char))
        return 'i';
    else if (t == typeid(short))
        return 'i';
    else if (t == typeid(long))
        return 'i';
    else if (t == typeid(long long))
        return 'i';

    else if (t == typeid(unsigned char))
        return 'u';
    else if (t == typeid(unsigned short))
        return 'u';
    else if (t == typeid(unsigned long))
        return 'u';
    else if (t == typeid(unsigned long long))
        return 'u';
    else if (t == typeid(unsigned int))
        return 'u';

    else if (t == typeid(bool))
        return 'b';

    else if (t == typeid(std::complex<float>))
        return 'c';
    else if (t == typeid(std::complex<double>))
        return 'c';
    else if (t == typeid(std::complex<long double>))
        return 'c';
    else if (t == typeid(std::string))
        return 'S';
    else
        return '?';
}

void
mqi::io::save_npz(std::string filename,
                  std::string var_name,
                  std::string data,
                  size_t      shape,
                  std::string mode) {
    std::fstream      fid_out;
    uint16_t          nrecs                = 0;
    size_t            global_header_offset = 0;
    std::vector<char> global_header;
    if (mode == "a") {
        size_t global_header_size;
        mqi::io::parse_zip_footer(filename, nrecs, global_header_size, global_header_offset);
        fid_out.open(filename, std::ios::in | std::ios::out | std::ios::binary | std::ios::ate);
        fid_out.seekg(0, fid_out.beg);
        if (fid_out.eof()) { printf("end of file reached\n"); }
        fid_out.seekg(global_header_offset, fid_out.beg);
        global_header.resize(global_header_size);
        if (fid_out.eof()) { printf("end of file reached\n"); }

        fid_out.read(&global_header[0], global_header_size);

        if (!fid_out) {
            std::cout << "error: only " << fid_out.gcount() << " could be read" << std::endl;
            exit(-1);
        }
        fid_out.seekg(global_header_offset, fid_out.beg);

    } else if (mode == "w") {
        fid_out.open(filename, std::ios::out | std::ios::binary);
    } else {
        throw std::runtime_error("Undefined mode\n");
    }
    std::vector<char> dict;
    mqi::io::push_value(dict, "{'descr': '");
    if (mqi::io::map_type(typeid(std::string)) == 'S') {
        mqi::io::push_value(dict, "|");
        dict.push_back(mqi::io::map_type(typeid(std::string)));
        mqi::io::push_value(dict, std::to_string(data.length()));
    } else {
        printf("wrong call\n");
        exit(-1);
    }

    mqi::io::push_value(dict, "', 'fortran_order': False, 'shape': (");
    mqi::io::push_value(dict, "), }");
    int remainder = 16 - (10 + dict.size()) % 16;
    dict.insert(dict.end(), remainder, ' ');
    dict.push_back('\n');
    std::vector<char> header;
    header.push_back((char) 0x93);
    mqi::io::push_value(header, "NUMPY");
    header.push_back((char) 0x01);
    header.push_back((char) 0x00);
    mqi::io::push_value(header, (uint16_t) dict.size());
    header.insert(header.end(), dict.begin(), dict.end());

    size_t   nbytes = sizeof(std::string) * data.length() + header.size();
    uint32_t crc    = crc32(0L, (uint8_t*) &header[0], header.size());
    crc             = crc32(crc, (uint8_t*) data.c_str(), sizeof(std::string) * data.length());

    //build the local header
    std::vector<char> local_header;
    mqi::io::push_value(local_header, "PK");                         //first part of sig
    mqi::io::push_value(local_header, (uint16_t) 0x0403);            //second part of sig
    mqi::io::push_value(local_header, (uint16_t) 20);                //min version to extract
    mqi::io::push_value(local_header, (uint16_t) 0);                 //general purpose bit flag
    mqi::io::push_value(local_header, (uint16_t) 0);                 //compression method
    mqi::io::push_value(local_header, (uint16_t) 0);                 //file last mod time
    mqi::io::push_value(local_header, (uint16_t) 0);                 //file last mod date
    mqi::io::push_value(local_header, (uint32_t) crc);               //crc
    mqi::io::push_value(local_header, (uint32_t) nbytes);            //compressed size
    mqi::io::push_value(local_header, (uint32_t) nbytes);            //uncompressed size
    mqi::io::push_value(local_header, (uint16_t) var_name.size());   //fname length
    mqi::io::push_value(local_header, (uint16_t) 0);                 //extra field length
    mqi::io::push_value(local_header, var_name);
    //build global header
    mqi::io::push_value(global_header, "PK");                //first part of sig
    mqi::io::push_value(global_header, (uint16_t) 0x0201);   //second part of sig
    mqi::io::push_value(global_header, (uint16_t) 20);       //version made by
    global_header.insert(global_header.end(), local_header.begin() + 4, local_header.begin() + 30);
    mqi::io::push_value(global_header, (uint16_t) 0);   //file comment length
    mqi::io::push_value(global_header, (uint16_t) 0);   //disk number where file starts
    mqi::io::push_value(global_header, (uint16_t) 0);   //internal file attributes
    mqi::io::push_value(global_header, (uint32_t) 0);   //external file attributes
    mqi::io::push_value(
      global_header,
      (uint32_t)
        global_header_offset);   //relative offset of local file header, since it begins where the global header used to begin
    mqi::io::push_value(global_header, var_name);
    std::vector<char> footer;
    mqi::io::push_value(footer, "PK");                              //first part of sig
    mqi::io::push_value(footer, (uint16_t) 0x0605);                 //second part of sig
    mqi::io::push_value(footer, (uint16_t) 0);                      //number of this disk
    mqi::io::push_value(footer, (uint16_t) 0);                      //disk where footer starts
    mqi::io::push_value(footer, (uint16_t) (nrecs + 1));            //number of records on this disk
    mqi::io::push_value(footer, (uint16_t) (nrecs + 1));            //total number of records
    mqi::io::push_value(footer, (uint32_t) global_header.size());   //nbytes of global headers
    mqi::io::push_value(
      footer,
      (uint32_t) (global_header_offset + nbytes +
                  local_header
                    .size()));   //offset of start of global headers, since global header now starts after newly written array
    mqi::io::push_value(footer, (uint16_t) 0);   //zip file comment length

    fid_out.write(reinterpret_cast<const char*>(&local_header[0]),
                  sizeof(char) * local_header.size());
    fid_out.write(reinterpret_cast<const char*>(&header[0]), sizeof(char) * header.size());
    fid_out.write(reinterpret_cast<const char*>(&data[0]), sizeof(std::string) * data.length());
    fid_out.write(reinterpret_cast<const char*>(&global_header[0]),
                  sizeof(char) * global_header.size());
    fid_out.write(reinterpret_cast<const char*>(&footer[0]), sizeof(char) * footer.size());
    fid_out.close();
}

template<typename T>
void
mqi::io::save_npz(std::string filename,
                  std::string var_name,
                  T*          data,
                  size_t      shape,
                  std::string mode) {
    std::fstream      fid_out;
    uint16_t          nrecs                = 0;
    size_t            global_header_offset = 0;
    std::vector<char> global_header;
    if (mode == "a") {
        size_t global_header_size;
        mqi::io::parse_zip_footer(filename, nrecs, global_header_size, global_header_offset);
        fid_out.open(filename, std::ios::in | std::ios::out | std::ios::binary | std::ios::ate);
        fid_out.seekg(0, fid_out.beg);
        if (fid_out.eof()) { printf("end of file reached\n"); }
        fid_out.seekg(global_header_offset, fid_out.beg);

        global_header.resize(global_header_size);
        if (fid_out.eof()) { printf("end of file reached\n"); }

        fid_out.read(&global_header[0], global_header_size);

        if (!fid_out) {
            std::cout << "error: only " << fid_out.gcount() << " could be read" << std::endl;
            exit(-1);
        }
        fid_out.seekg(global_header_offset, fid_out.beg);

    } else if (mode == "w") {
        fid_out.open(filename, std::ios::out | std::ios::binary);
    } else {
        throw std::runtime_error("Undefined mode\n");
    }
    std::vector<char> dict;
    mqi::io::push_value(dict, "{'descr': '");
    if (mqi::io::map_type(typeid(T)) == 'S') {
        mqi::io::push_value(dict, "|");
        dict.push_back(mqi::io::map_type(typeid(T)));
        mqi::io::push_value(dict, "4");
    } else {
        mqi::io::push_value(dict, "<");
        dict.push_back(mqi::io::map_type(typeid(T)));
        mqi::io::push_value(dict, std::to_string(sizeof(T)));
    }

    mqi::io::push_value(dict, "', 'fortran_order': False, 'shape': (");
    mqi::io::push_value(dict, std::to_string(shape));
    mqi::io::push_value(dict, ",), }");
    int remainder = 16 - (10 + dict.size()) % 16;
    dict.insert(dict.end(), remainder, ' ');
    dict.push_back('\n');
    std::vector<char> header;
    header.push_back((char) 0x93);
    mqi::io::push_value(header, "NUMPY");
    header.push_back((char) 0x01);
    header.push_back((char) 0x00);
    mqi::io::push_value(header, (uint16_t) dict.size());
    header.insert(header.end(), dict.begin(), dict.end());

    size_t   nbytes = shape * sizeof(T) + header.size();
    uint32_t crc    = crc32(0L, (uint8_t*) &header[0], header.size());
    crc             = crc32(crc, (uint8_t*) data, shape * sizeof(T));

    //build the local header
    std::vector<char> local_header;
    mqi::io::push_value(local_header, "PK");                         //first part of sig
    mqi::io::push_value(local_header, (uint16_t) 0x0403);            //second part of sig
    mqi::io::push_value(local_header, (uint16_t) 20);                //min version to extract
    mqi::io::push_value(local_header, (uint16_t) 0);                 //general purpose bit flag
    mqi::io::push_value(local_header, (uint16_t) 0);                 //compression method
    mqi::io::push_value(local_header, (uint16_t) 0);                 //file last mod time
    mqi::io::push_value(local_header, (uint16_t) 0);                 //file last mod date
    mqi::io::push_value(local_header, (uint32_t) crc);               //crc
    mqi::io::push_value(local_header, (uint32_t) nbytes);            //compressed size
    mqi::io::push_value(local_header, (uint32_t) nbytes);            //uncompressed size
    mqi::io::push_value(local_header, (uint16_t) var_name.size());   //fname length
    mqi::io::push_value(local_header, (uint16_t) 0);                 //extra field length
    mqi::io::push_value(local_header, var_name);
    //build global header
    mqi::io::push_value(global_header, "PK");                //first part of sig
    mqi::io::push_value(global_header, (uint16_t) 0x0201);   //second part of sig
    mqi::io::push_value(global_header, (uint16_t) 20);       //version made by
    global_header.insert(global_header.end(), local_header.begin() + 4, local_header.begin() + 30);
    mqi::io::push_value(global_header, (uint16_t) 0);   //file comment length
    mqi::io::push_value(global_header, (uint16_t) 0);   //disk number where file starts
    mqi::io::push_value(global_header, (uint16_t) 0);   //internal file attributes
    mqi::io::push_value(global_header, (uint32_t) 0);   //external file attributes
    mqi::io::push_value(
      global_header,
      (uint32_t)
        global_header_offset);   //relative offset of local file header, since it begins where the global header used to begin
    mqi::io::push_value(global_header, var_name);
    //build footer
    std::vector<char> footer;
    mqi::io::push_value(footer, "PK");                              //first part of sig
    mqi::io::push_value(footer, (uint16_t) 0x0605);                 //second part of sig
    mqi::io::push_value(footer, (uint16_t) 0);                      //number of this disk
    mqi::io::push_value(footer, (uint16_t) 0);                      //disk where footer starts
    mqi::io::push_value(footer, (uint16_t) (nrecs + 1));            //number of records on this disk
    mqi::io::push_value(footer, (uint16_t) (nrecs + 1));            //total number of records
    mqi::io::push_value(footer, (uint32_t) global_header.size());   //nbytes of global headers
    mqi::io::push_value(
      footer,
      (uint32_t) (global_header_offset + nbytes +
                  local_header
                    .size()));   //offset of start of global headers, since global header now starts after newly written array
    mqi::io::push_value(footer, (uint16_t) 0);   //zip file comment length

    fid_out.write(reinterpret_cast<const char*>(&local_header[0]),
                  sizeof(char) * local_header.size());
    fid_out.write(reinterpret_cast<const char*>(&header[0]), sizeof(char) * header.size());
    fid_out.write(reinterpret_cast<const char*>(&data[0]), sizeof(T) * shape);
    fid_out.write(reinterpret_cast<const char*>(&global_header[0]),
                  sizeof(char) * global_header.size());
    fid_out.write(reinterpret_cast<const char*>(&footer[0]), sizeof(char) * footer.size());
    fid_out.close();
}

#endif