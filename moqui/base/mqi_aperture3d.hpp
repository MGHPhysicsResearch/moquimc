#ifndef MQI_APERTURE3D_H
#define MQI_APERTURE3D_H

/// \file
///
/// Rectlinear grid geometry for MC transport
///

#include <moqui/base/mqi_common.hpp>
#include <moqui/base/mqi_coordinate_transform.hpp>
#include <moqui/base/mqi_grid3d.hpp>
#include <moqui/base/mqi_math.hpp>
#include <moqui/base/mqi_vec.hpp>
//typedef float phsp_t;

namespace mqi
{
/// \class aperture3d
/// \tparam T for grid values, e.g., dose, HU, vector
/// \tparam R for grid coordinates, float, double, etc.
template<typename T, typename R>
class aperture3d : public grid3d<T, R>
{
public:
    uint16_t       num_opening;
    uint16_t*      num_segments;
    mqi::vec2<R>** block_segment;
    //    mqi::mat3x3<R> rotation_matrix_fwd;
    //    mqi::mat3x3<R> rotation_matrix_inv;
    //    mqi::vec3<R>   translation_vector;
    ///< Default constructor only for child classes
    ///cuda_host_device or cuda_host
    /// note (Feb27,2020): it may be uesless
    CUDA_HOST_DEVICE
    aperture3d() : grid3d<T, R>() {
        ;
    }

    /// Construct a rectlinear grid from array of x/y/z with their size
    /// \param x,y,z  1D array of central points of voxels along x-axis
    /// \param xn,yn,zn  size of 1D array for points.
    CUDA_HOST_DEVICE
    aperture3d(const R     xe[],
               const ijk_t n_xe,
               const R     ye[],
               const ijk_t n_ye,
               const R     ze[],
               const ijk_t n_ze) :
        grid3d<T, R>(xe, n_xe, ye, n_ye, ze, n_ze) {}

    /// Construct a rectlinear grid from array of x/y/z with their size
    /// \param x,y,z  1D array of central points of voxels along x-axis
    /// \param xn,yn,zn  size of 1D array for points.
    CUDA_HOST_DEVICE
    aperture3d(const R     xe_min,
               const R     xe_max,
               const ijk_t n_xe,   //n_xe : steps + 1
               const R     ye_min,
               const R     ye_max,
               const ijk_t n_ye,
               const R     ze_min,
               const R     ze_max,
               const ijk_t n_ze) :
        grid3d<T, R>(xe_min, xe_max, n_xe, ye_min, ye_max, n_ye, ze_min, ze_max, n_ze) {}

    /// Constructor for oriented bounding boxess
    /// Construct a rectlinear grid from array of x/y/z with their size and rotation angless
    /// \param x,y,z  1D array of central points of voxels along x-axis
    /// \param xn,yn,zn  size of 1D array for points.
    /// \parmas angles rotation angle in degree for each axis.
    CUDA_HOST_DEVICE
    aperture3d(const R           xe_min,
               const R           xe_max,
               const ijk_t       n_xe,   //n_xe : steps + 1
               const R           ye_min,
               const R           ye_max,
               const ijk_t       n_ye,
               const R           ze_min,
               const R           ze_max,
               const ijk_t       n_ze,
               std::array<R, 3>& angles) :
        grid3d<T, R>(xe_min, xe_max, n_xe, ye_min, ye_max, n_ye, ze_min, ze_max, n_ze, angles) {}

    /// Constructor for oriented bounding boxess
    /// Construct a rectlinear grid from array of x/y/z with their size and rotation angless
    /// \param x,y,z  1D array of central points of voxels along x-axis
    /// \param xn,yn,zn  size of 1D array for points.
    /// \parmas angles rotation angle in degree for each axis.
    CUDA_HOST_DEVICE
    aperture3d(const R        xe_min,
               const R        xe_max,
               const ijk_t    n_xe,   //n_xe : steps + 1
               const R        ye_min,
               const R        ye_max,
               const ijk_t    n_ye,
               const R        ze_min,
               const R        ze_max,
               const ijk_t    n_ze,
               mqi::mat3x3<R> rxyz) :
        grid3d<T, R>(xe_min, xe_max, n_xe, ye_min, ye_max, n_ye, ze_min, ze_max, n_ze, rxyz) {}

    /// Construct a rectlinear grid from array of x/y/z with their size
    /// \param x,y,z  1D array of central points of voxels along x-axis
    /// \param xn,yn,zn  size of 1D array for points.
    CUDA_HOST_DEVICE
    aperture3d(const R        xe[],
               const ijk_t    n_xe,
               const R        ye[],
               const ijk_t    n_ye,
               const R        ze[],
               const ijk_t    n_ze,
               mqi::mat3x3<R> rxyz) :
        grid3d<T, R>(xe, n_xe, ye, n_ye, ze, n_ze, rxyz) {}

    /// Construct a rectlinear grid from array of x/y/z with their size
    /// \param x,y,z  1D array of central points of voxels along x-axis
    /// \param xn,yn,zn  size of 1D array for points.
    CUDA_HOST_DEVICE
    aperture3d(const R           xe[],
               const ijk_t       n_xe,
               const R           ye[],
               const ijk_t       n_ye,
               const R           ze[],
               const ijk_t       n_ze,
               std::array<R, 3>& angles) :
        grid3d<T, R>(xe, n_xe, ye, n_ye, ze, n_ze) {}

    /// Constructor for oriented bounding boxess
    /// Construct a rectlinear grid from array of x/y/z with their size and rotation angless
    /// \param x,y,z  1D array of central points of voxels along x-axis
    /// \param xn,yn,zn  size of 1D array for points.
    /// \parmas angles rotation angle in degree for each axis.
    CUDA_HOST_DEVICE
    aperture3d(const R           xe_min,
               const R           xe_max,
               const ijk_t       n_xe,   //n_xe : steps + 1
               const R           ye_min,
               const R           ye_max,
               const ijk_t       n_ye,
               const R           ze_min,
               const R           ze_max,
               const ijk_t       n_ze,
               std::array<R, 3>& angles,
               int16_t           num_opening,
               uint16_t*         num_segment,
               mqi::vec2<R>**    block_segment) :
        grid3d<T, R>(xe_min, xe_max, n_xe, ye_min, ye_max, n_ye, ze_min, ze_max, n_ze, angles) {
        this->num_opening   = num_opening;
        this->num_segments  = num_segment;
        this->block_segment = block_segment;
    }

    ///< Destructor releases dynamic allocation for x/y/z coordinates
    CUDA_HOST_DEVICE
    ~aperture3d() {
        /*
            delete[] xe_;
            delete[] ye_;
            delete[] ze_;
            */
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
    is_inside(mqi::vec3<R> pos) {
        //    printf("block data size %lu\n", block_data.size());
        bool inside;
        for (int i = 0; i < this->num_opening; i++) {
            mqi::vec2<R>* segment = this->block_segment[i];
            //        std::vector<std::array<float, 2>> segment = block_data[i];
            //        printf("segment size %lu\n", segment.size());
            //        inside = sol1(pos, segment, volume->num_segment[i]);
            inside = sol1_1(pos, segment, this->num_segments[i]);
        }
        return inside;
    }

    ///< intersect. a ray from a voxel (ijk) in the grid
    /// written by Hoyeon, plane equation based
    CUDA_HOST_DEVICE
    virtual intersect_t<R>
    intersect(mqi::vec3<R>& p, mqi::vec3<R>& d, mqi::vec3<ijk_t> idx) {
        //        printf("in ");
        //        idx.dump();
        /// n100_ is vector of x-axis
        /// Change the method to operate with roated box
        mqi::intersect_t<R> its;   //return value
        its.cell = idx;
        its.side = mqi::NONE_XYZ_PLANE;
        mqi::intersect_t<R> its_out;   //return value
        mqi::intersect_t<R> its_in;    //return value
        its_out.dist   = 0;
        its_out.cell.x = -5;
        its_out.cell.y = -5;
        its_out.cell.z = -5;
        its_in.dist    = 0;
        its_in.cell.x  = -10;
        its_in.cell.y  = -10;
        its_in.cell.z  = -10;
        //        printf("in ");
        //        idx.dump();
        if (!is_inside(p)) {
            //            printf("aperture out ");
            //            its_out.cell.dump();
            its_out.type = mqi::APERTURE_CLOSE;
            return its_out;
        } else {
            if (d.z < 0) {
                if (mqi::mqi_abs(p.z - this->ze_[0]) < 1e-3) {
                    its_in.dist = 0.0;
                } else {
                    its_in.dist = -(p.z - this->ze_[0]) / d.z;
                }

            } else if (d.z > 0) {
                if (mqi::mqi_abs(p.z - this->ze_[1]) < 1e-3) {
                    its_in.dist = 0;
                } else {
                    its_in.dist = (this->ze_[1] - p.z) / d.z;
                }
            } else {
                its_in.dist = 0;
            }
            assert(its_in.dist >= 0);
            //            printf("aperture in ");
            //            its_in.cell.dump();
            its_in.type = mqi::APERTURE_OPEN;
            return its_in;
        }
    }
};

}   // namespace mqi

#endif
