#ifndef MQI_GRID3D_H
#define MQI_GRID3D_H

/// \file
///
/// Rectlinear grid geometry for MC transport
///

#include <moqui/base/mqi_common.hpp>
#include <moqui/base/mqi_coordinate_transform.hpp>
#include <moqui/base/mqi_math.hpp>
#include <moqui/base/mqi_vec.hpp>
//typedef double phsp_t;

namespace mqi
{

///< struct to describe intersect information
///  plane is between v0 ~ v1 ?
template<typename R>
struct intersect_t {
    R              dist;   //distance to the intersection plane (cell), invalid when dist < 0
    cell_side      side;   //side of entering cell
    vec3<ijk_t>    cell;   //index of entering cell
    transport_type type;   // type of current node's geometry
};

/// \class grid3d
/// \tparam T for grid values, e.g., dose, HU, vector
/// \tparam R for grid coordinates, float, double, etc.
template<typename T, typename R>
class grid3d
{

protected:
    ///< Number of data points (pixels) not edges.
    /// dim_.x, dim_.y, dim_.z
    mqi::vec3<ijk_t> dim_;

    /// the edge positions of pixels
    /// Bounding box xe_[0], xe_[max]
    /// number of edge points is dim_.x/y/z + 1
    /// acsending order
    /// TODO: const R* xe_ ?
    R* xe_ = nullptr;   ///< x_min, ..., x_max
    R* ye_ = nullptr;   ///< y_min, ..., y_max
    R* ze_ = nullptr;   ///< z_min, ..., z_max

    /// Two edge position of bounding box
    mqi::vec3<R> V000_;   ///< coner x_min, y_min, z_min
    mqi::vec3<R> V111_;   ///< coner x_max, y_max, z_max
    mqi::vec3<R> C_;      ///< Center

    mqi::vec3<R> n100_;   ///< normal vector of 1st (X) axis
    mqi::vec3<R> n010_;   ///< normal vector of 2nd (Y) axis
    mqi::vec3<R> n001_;   ///< normal vector of 3rd (Z) axis

    ///< Data in this rectlinear
    ///< size: dim_.x*dim_.y*dim_.z
    T* data_ = nullptr;

    ///< Calculate C000/C111
    CUDA_HOST_DEVICE
    void
    calculate_bounding_box(void) {
        V000_.x = xe_[0];
        V000_.y = ye_[0];
        V000_.z = ze_[0];
        V111_.x = xe_[dim_.x];
        V111_.y = ye_[dim_.y];
        V111_.z = ze_[dim_.z];

        C_ = (V000_ + V111_) * 0.5;

        n100_.x = V111_.x - V000_.x;
        n100_.y = 0.0;
        n100_.z = 0.0;

        n010_.x = 0.0;
        n010_.y = V111_.y - V000_.y;
        n010_.z = 0.0;

        n001_.x = 0.0;
        n001_.y = 0.0;
        n001_.z = V111_.z - V000_.z;
        n100_.normalize();
        n010_.normalize();
        n001_.normalize();
    }

public:
    mqi::mat3x3<R> rotation_matrix_fwd;
    mqi::mat3x3<R> rotation_matrix_inv;
    mqi::vec3<R>   translation_vector;
    ///< Default constructor only for child classes
    ///cuda_host_device or cuda_host
    /// note (Feb27,2020): it may be uesless
    CUDA_HOST_DEVICE
    grid3d() {
        ;
    }

    /// Construct a rectlinear grid from array of x/y/z with their size
    /// \param x,y,z  1D array of central points of voxels along x-axis
    /// \param xn,yn,zn  size of 1D array for points.
    CUDA_HOST_DEVICE
    grid3d(const R     xe[],
           const ijk_t n_xe,
           const R     ye[],
           const ijk_t n_ye,
           const R     ze[],
           const ijk_t n_ze) {
        xe_ = new R[n_xe];
        ye_ = new R[n_ye];
        ze_ = new R[n_ze];

        for (ijk_t i = 0; i < n_xe; ++i)
            xe_[i] = xe[i];
        for (ijk_t i = 0; i < n_ye; ++i)
            ye_[i] = ye[i];
        for (ijk_t i = 0; i < n_ze; ++i)
            ze_[i] = ze[i];

        dim_.x = n_xe - 1;
        dim_.y = n_ye - 1;
        dim_.z = n_ze - 1;

        this->calculate_bounding_box();
    }

    /// Construct a rectlinear grid from array of x/y/z with their size
    /// \param x,y,z  1D array of central points of voxels along x-axis
    /// \param xn,yn,zn  size of 1D array for points.
    CUDA_HOST_DEVICE
    grid3d(const R     xe_min,
           const R     xe_max,
           const ijk_t n_xe,   //n_xe : steps + 1
           const R     ye_min,
           const R     ye_max,
           const ijk_t n_ye,
           const R     ze_min,
           const R     ze_max,
           const ijk_t n_ze) {
        xe_ = new R[n_xe];
        ye_ = new R[n_ye];
        ze_ = new R[n_ze];

        dim_.x = n_xe - 1;
        dim_.y = n_ye - 1;
        dim_.z = n_ze - 1;

        const R dx = (xe_max - xe_min) / dim_.x;
        const R dy = (ye_max - ye_min) / dim_.y;
        const R dz = (ze_max - ze_min) / dim_.z;

        for (ijk_t i = 0; i < n_xe; ++i)
            xe_[i] = xe_min + i * dx;
        for (ijk_t i = 0; i < n_ye; ++i)
            ye_[i] = ye_min + i * dy;
        for (ijk_t i = 0; i < n_ze; ++i)
            ze_[i] = ze_min + i * dz;

        this->calculate_bounding_box();
    }

    /// Constructor for oriented bounding boxess
    /// Construct a rectlinear grid from array of x/y/z with their size and rotation angless
    /// \param x,y,z  1D array of central points of voxels along x-axis
    /// \param xn,yn,zn  size of 1D array for points.
    /// \parmas angles rotation angle in degree for each axis.
    CUDA_HOST_DEVICE
    grid3d(const R           xe_min,
           const R           xe_max,
           const ijk_t       n_xe,   //n_xe : steps + 1
           const R           ye_min,
           const R           ye_max,
           const ijk_t       n_ye,
           const R           ze_min,
           const R           ze_max,
           const ijk_t       n_ze,
           std::array<R, 3>& angles) {
        xe_ = new R[n_xe];
        ye_ = new R[n_ye];
        ze_ = new R[n_ze];

        dim_.x = n_xe - 1;
        dim_.y = n_ye - 1;
        dim_.z = n_ze - 1;

        const R dx = (xe_max - xe_min) / dim_.x;
        const R dy = (ye_max - ye_min) / dim_.y;
        const R dz = (ze_max - ze_min) / dim_.z;

        for (ijk_t i = 0; i < n_xe; ++i)
            xe_[i] = xe_min + i * dx;
        for (ijk_t i = 0; i < n_ye; ++i)
            ye_[i] = ye_min + i * dy;
        for (ijk_t i = 0; i < n_ze; ++i)
            ze_[i] = ze_min + i * dz;
        this->calculate_bounding_box();
        /// Define rotation matrix
        /// The rotation matrix is to rotate the other objects
        /// to have the same geometric position to the bouding box

        rotation_matrix_fwd.rotate(static_cast<R>(angles[0] * M_PI / 180.0),
                                   static_cast<R>(angles[1] * M_PI / 180.0),
                                   static_cast<R>(angles[2] * M_PI / 180.0));
        rotation_matrix_inv = rotation_matrix_fwd.inverse();
    }

    /// Constructor for oriented bounding boxess
    /// Construct a rectlinear grid from array of x/y/z with their size and rotation angless
    /// \param x,y,z  1D array of central points of voxels along x-axis
    /// \param xn,yn,zn  size of 1D array for points.
    /// \parmas angles rotation angle in degree for each axis.
    CUDA_HOST_DEVICE
    grid3d(const R        xe_min,
           const R        xe_max,
           const ijk_t    n_xe,   //n_xe : steps + 1
           const R        ye_min,
           const R        ye_max,
           const ijk_t    n_ye,
           const R        ze_min,
           const R        ze_max,
           const ijk_t    n_ze,
           mqi::mat3x3<R> rxyz) {
        xe_ = new R[n_xe];
        ye_ = new R[n_ye];
        ze_ = new R[n_ze];

        dim_.x = n_xe - 1;
        dim_.y = n_ye - 1;
        dim_.z = n_ze - 1;

        const R dx = (xe_max - xe_min) / dim_.x;
        const R dy = (ye_max - ye_min) / dim_.y;
        const R dz = (ze_max - ze_min) / dim_.z;

        for (ijk_t i = 0; i < n_xe; ++i)
            xe_[i] = xe_min + i * dx;
        for (ijk_t i = 0; i < n_ye; ++i)
            ye_[i] = ye_min + i * dy;
        for (ijk_t i = 0; i < n_ze; ++i)
            ze_[i] = ze_min + i * dz;
        this->calculate_bounding_box();
        /// Define rotation matrix
        /// The rotation matrix is to rotate the other objects
        /// to have the same geometric position to the bouding box
        rotation_matrix_fwd = rxyz;
        rotation_matrix_inv = rotation_matrix_fwd.inverse();
    }

    /// Construct a rectlinear grid from array of x/y/z with their size
    /// \param x,y,z  1D array of central points of voxels along x-axis
    /// \param xn,yn,zn  size of 1D array for points.
    CUDA_HOST_DEVICE
    grid3d(const R        xe[],
           const ijk_t    n_xe,
           const R        ye[],
           const ijk_t    n_ye,
           const R        ze[],
           const ijk_t    n_ze,
           mqi::mat3x3<R> rxyz) {
        xe_ = new R[n_xe];
        ye_ = new R[n_ye];
        ze_ = new R[n_ze];

        for (ijk_t i = 0; i < n_xe; ++i)
            xe_[i] = xe[i];
        for (ijk_t i = 0; i < n_ye; ++i)
            ye_[i] = ye[i];
        for (ijk_t i = 0; i < n_ze; ++i)
            ze_[i] = ze[i];

        dim_.x = n_xe - 1;
        dim_.y = n_ye - 1;
        dim_.z = n_ze - 1;

        this->calculate_bounding_box();

        rotation_matrix_fwd = rxyz;
        rotation_matrix_inv = rotation_matrix_fwd.inverse();
    }

    /// Construct a rectlinear grid from array of x/y/z with their size
    /// \param x,y,z  1D array of central points of voxels along x-axis
    /// \param xn,yn,zn  size of 1D array for points.
    CUDA_HOST_DEVICE
    grid3d(const R           xe[],
           const ijk_t       n_xe,
           const R           ye[],
           const ijk_t       n_ye,
           const R           ze[],
           const ijk_t       n_ze,
           std::array<R, 3>& angles) {
        xe_ = new R[n_xe];
        ye_ = new R[n_ye];
        ze_ = new R[n_ze];

        for (ijk_t i = 0; i < n_xe; ++i)
            xe_[i] = xe[i];
        for (ijk_t i = 0; i < n_ye; ++i)
            ye_[i] = ye[i];
        for (ijk_t i = 0; i < n_ze; ++i)
            ze_[i] = ze[i];

        dim_.x = n_xe - 1;
        dim_.y = n_ye - 1;
        dim_.z = n_ze - 1;

        this->calculate_bounding_box();

        rotation_matrix_fwd.rotate(static_cast<R>(angles[0] * M_PI / 180.0),
                                   static_cast<R>(angles[1] * M_PI / 180.0),
                                   static_cast<R>(angles[2] * M_PI / 180.0));
        rotation_matrix_inv = rotation_matrix_fwd.inverse();
    }

    ///< Destructor releases dynamic allocation for x/y/z coordinates
    CUDA_HOST_DEVICE
    ~grid3d() {}

    /// set edge pointer
    /// \return pointer of data
    CUDA_HOST_DEVICE
    virtual void
    set_edges(R* xe, ijk_t nx, R* ye, ijk_t ny, R* ze, ijk_t nz) {
        xe_    = xe;
        ye_    = ye;
        ze_    = ze;
        dim_.x = nx - 1;
        dim_.y = ny - 1;
        dim_.z = nz - 1;
        this->calculate_bounding_box();
    }

    CUDA_HOST_DEVICE
    virtual R*
    get_x_edges() {
        return xe_;
    }

    CUDA_HOST_DEVICE
    virtual R*
    get_y_edges() {
        return ye_;
    }

    CUDA_HOST_DEVICE
    virtual R*
    get_z_edges() {
        return ze_;
    }

    /// Returns number of bins box
    CUDA_HOST_DEVICE
    mqi::vec3<ijk_t>
    get_nxyz() {
        return dim_;
    }

    /// Returns the data value for given x/y/z index
    /// \param[in] p index, p[0], p[1], p[2] for x, y, z.
    CUDA_HOST_DEVICE
    virtual const T
    operator[](const mqi::vec3<ijk_t> p) {
        return data_[ijk2cnb(p.x, p.y, p.z)];
    }

    /// Returns the data value for given x/y/z index
    /// \param[in] p index, p[0], p[1], p[2] for x, y, z.
    CUDA_HOST_DEVICE
    virtual const T
    operator[](const mqi::cnb_t p) {
        return data_[p];
    }

    /// Prints out x,y,z coordinate positions
    CUDA_HOST_DEVICE
    virtual void
    dump_edges() {
        printf("X edges: ");
        for (ijk_t i = 0; i <= dim_.x; ++i) {
            printf(" %f", xe_[i]);
        }
        printf("\n");

        printf("Y edges: ");
        for (ijk_t i = 0; i <= dim_.y; ++i) {
            printf(" %f", ye_[i]);
        }
        printf("\n");

        printf("Z edges: ");
        for (ijk_t i = 0; i <= dim_.z; ++i) {
            printf(" %f", ze_[i]);
        }
        printf("\n");
    }

    /// Converts index of x,y,z to index of valarray(data)
    CUDA_HOST_DEVICE
    virtual inline cnb_t
    ijk2cnb(ijk_t i, ijk_t j, ijk_t k) {
        return k * dim_.x * dim_.y + j * dim_.x + i;
    }

    /// Converts index of x,y,z to index of valarray(data)
    CUDA_HOST_DEVICE
    cnb_t
    ijk2cnb(vec3<ijk_t> idx) {
        return idx.z * dim_.x * dim_.y + idx.y * dim_.x + idx.x;
    }

    /// Converts copy number to index of x,y,z
    CUDA_HOST_DEVICE
    virtual inline vec3<ijk_t>
    cnb2ijk(cnb_t c) {
        const cnb_t nxy = dim_.x * dim_.y;
        vec3<ijk_t> ijk;
        ijk.z = c / nxy;
        ijk.y = (c % (nxy)) / dim_.x;
        ijk.x = (c % (nxy)) % dim_.x;
        //ijk.dump();
        return ijk;
    }

    CUDA_HOST_DEVICE
    void
    delete_data_if_used(void) {
        if (data_ != nullptr) delete[] data_;
    }

    /// Initializes data, currently values are sum of index square for testing
    CUDA_HOST
    virtual void
    load_data() {
        //this->delete_data_if_used();
        //data_ = new T[dim_.x * dim_.y * dim_.z];
    }

    /// Reads data from other source,  currently values are sum of index square for testing
    /// //total == dim_.x*dim_.y*dim_.z : not sure total is useful.
    /// //will copy
    /// in case src is CPU and dest GPU
    CUDA_HOST_DEVICE
    virtual void
    set_data(T* src) {
        this->delete_data_if_used();
        data_ = src;   //can be change pointer but we will copy.
    }

    /// Fills data with a given value
    CUDA_HOST_DEVICE
    virtual void
    fill_data(T a) {
        data_ = new T[dim_.x * dim_.y * dim_.z];
        for (uint32_t i = 0; i < dim_.x * dim_.y * dim_.z; ++i)
            data_[i] = a;
    }

    /// Returns data
    /// \return pointer of data
    CUDA_HOST_DEVICE
    T*
    get_data() const {
        return data_;
    }

    CUDA_HOST_DEVICE
    R
    get_volume(const mqi::cnb_t p) {
        vec3<ijk_t> vox    = cnb2ijk(p);
        R           volume = xe_[vox.x + 1] - xe_[vox.x];
        volume *= ye_[vox.y + 1] - ye_[vox.y];
        volume *= ze_[vox.z + 1] - ze_[vox.z];
        return volume;
    }

    CUDA_HOST_DEVICE
    R
    get_volume(const mqi::vec3<ijk_t> vox) {
        R volume = xe_[vox.x + 1] - xe_[vox.x];
        volume *= ye_[vox.y + 1] - ye_[vox.y];
        volume *= ze_[vox.z + 1] - ze_[vox.z];
        return volume;
    }
    ///< intersect. a ray from a voxel (ijk) in the grid
    /// written by Hoyeon, plane equation based
    CUDA_HOST_DEVICE
    intersect_t<R>
    intersect(mqi::vec3<R>& p, mqi::vec3<R>& d, mqi::vec3<ijk_t>& idx) {
        /// n100_ is vector of x-axis
        /// Change the method to operate with roated box
        mqi::intersect_t<R> its;   //return value
        its.cell = idx;
        its.side = mqi::NONE_XYZ_PLANE;
        mqi::intersect_t<R> its_non;   //return value
        its_non.side   = mqi::NONE_XYZ_PLANE;
        its_non.dist   = -1.0;
        its_non.cell.x = -1;
        its_non.cell.y = -1;
        its_non.cell.z = -1;
        ///< temporal variables
        mqi::vec3<R>         t_min, t_max;
        mqi::vec3<cell_side> side(NONE_XYZ_PLANE, NONE_XYZ_PLANE, NONE_XYZ_PLANE);
        mqi::vec3<R>         vox1;
        mqi::vec3<R>         vox2;
        vox1.x = xe_[idx.x];
        vox1.y = ye_[idx.y];
        vox1.z = ze_[idx.z];
        vox2.x = xe_[idx.x + 1];
        vox2.y = ye_[idx.y + 1];
        vox2.z = ze_[idx.z + 1];
        assert((vox1.x < p.x || fabsf(vox1.x - p.x) < 1e-3) &&
               (p.x < vox2.x || fabsf(vox2.x - p.x) < 1e-3));
        assert((vox1.y < p.y || fabsf(vox1.y - p.y) < 1e-3) &&
               (p.y < vox2.y || fabsf(vox2.y - p.y) < 1e-3));
        if ((vox1.z < p.z || fabsf(vox1.z - p.z) < 1e-3) &&
            (p.z < vox2.z || fabsf(vox2.z - p.z) < 1e-3)) {

        } else {
            printf("vtx1.z %f p.x %f vox1.z %f d.z %f\n", vox1.z, p.z, vox2.z, d.z);
        }
        assert((vox1.z < p.z || fabsf(vox1.z - p.z) < 1e-3) &&
               (p.z < vox2.z || fabsf(vox2.z - p.z) < 1e-3));

        ///< check X (1st axis)
        R me = d.dot(n100_);
        R pe = p.dot(n100_);
        /// non intersect
        /// Is this required? Since the particle is alreay in a voxel, it should intersect with some surface
        if (me * me > mqi::near_zero) {
            if (me < 0) {   //if intersect, X-
                if (mqi::mqi_abs(-(p.x - vox1.x) / d.x) < mqi::geometry_tolerance && idx.x > 0) {
                    t_max.x = 1 / mqi::geometry_tolerance;
                } else {
                    t_max.x = -(p.x - vox1.x) / d.x;
                }
                t_min.x = -(p.x - vox2.x) / d.x;
            } else {   //if intersect, X+
                t_min.x = (vox1.x - p.x) / d.x;
                if (mqi::mqi_abs((vox2.x - p.x) / d.x) < mqi::geometry_tolerance &&
                    idx.x < dim_.x) {
                    t_max.x = 1 / mqi::geometry_tolerance;
                } else {
                    t_max.x = (vox2.x - p.x) / d.x;
                }
            }
        } else {
            d.x     = 0;
            t_min.x = mqi_m_inf;
            t_max.x = mqi::p_inf;
        }   //----< X

        ///< check Y axis
        me = d.dot(n010_);
        pe = p.dot(n010_);
        if (me * me > mqi::near_zero) {
            if (me < 0) {   //if intersect, Y-
                if (mqi::mqi_abs(-(p.y - vox1.y) / d.y) < mqi::geometry_tolerance && idx.y > 0) {
                    t_max.y = 1 / mqi::geometry_tolerance;
                } else {
                    t_max.y = -(p.y - vox1.y) / d.y;
                }
                t_min.y = -(p.y - vox2.y) / d.y;
            } else {   //if intersect, Y+
                t_min.y = (vox1.y - p.y) / d.y;
                if (mqi::mqi_abs((vox2.y - p.y) / d.y) < mqi::geometry_tolerance &&
                    idx.y < dim_.y) {
                    t_max.y = 1 / mqi::geometry_tolerance;
                } else {
                    t_max.y = (vox2.y - p.y) / d.y;
                }
            }
        } else {
            d.y     = 0;
            t_min.y = mqi_m_inf;
            t_max.y = mqi::p_inf;
        }   //----< Y
        ///< check Z (3rd) axis
        me = d.dot(n001_);
        pe = p.dot(n001_);
        if (me * me > mqi::near_zero) {
            if (me < 0) {   //if intersect, Z-
                if (mqi::mqi_abs(-(p.z - vox1.z) / d.z) < mqi::geometry_tolerance && idx.z > 0) {
                    t_max.z = 1 / mqi::geometry_tolerance;
                } else {
                    t_max.z = -(p.z - vox1.z) / d.z;
                }
                t_min.z = -(p.z - vox2.z) / d.z;
            } else {   //if intersect, Z+
                t_min.z = (vox1.z - p.z) / d.z;
                if (mqi::mqi_abs((vox2.z - p.z) / d.z) < mqi::geometry_tolerance &&
                    idx.z < dim_.z) {
                    t_max.z = 1 / mqi::geometry_tolerance;
                } else {
                    t_max.z = (vox2.z - p.z) / d.z;
                }
            }
        } else {
            d.z     = 0;
            t_min.z = mqi_m_inf;
            t_max.z = mqi::p_inf;
        }

        ///< Find max value among T_min of x,y,z axis

        ///< Find min value among T_max of x,y,z axis
        R u_max;
        if (t_max.x < t_max.y) {
            u_max = (t_max.x < t_max.z) ? t_max.x : t_max.z;
        } else {
            u_max = (t_max.y < t_max.z) ? t_max.y : t_max.z;
        }

        ////            if (u_min < u_max && u_max >= 0) {
        if (u_max > 0) {
            its.dist = u_max;

        } else {

            return its_non;
        }
        return its;
    }

    ///< intersect. a ray from outside to entering the grid
    ///< return distance and side
    /// Calculated distance become integer and distance smaller than 1 is ignored
    CUDA_HOST_DEVICE
    intersect_t<R>
    intersect(mqi::vec3<R>& p, mqi::vec3<R>& d) {
        mqi::intersect_t<R> its;   //return value
        if (p.x >= xe_[0] && p.x <= xe_[dim_.x] && p.y >= ye_[0] && p.y <= ye_[dim_.y] &&
            p.z >= ze_[0] && p.z <= ze_[dim_.z]) {
            its.cell = this->index(p, d);
            its.dist = 0;
            its.side = mqi::NONE_XYZ_PLANE;
            return its;
        } else {
            its.dist   = -1.0;
            its.side   = mqi::NONE_XYZ_PLANE;
            its.cell.x = -1;
            its.cell.y = -1;
            its.cell.z = -1;
        }
        ///< temporal variables
        mqi::vec3<R>         t_min;
        mqi::vec3<R>         t_max;
        mqi::vec3<cell_side> side(XM, YM, ZM);
        mqi::vec3<R>         C2p = (p + d) - C_;
        C2p.normalize();

        ///< check X (1st axis)
        R me = d.dot(n100_);
        R pe = p.dot(n100_);

        /// non intersect
        if (me * me > mqi::near_zero) {
            if (me > 0) {                          //if intersect, X+
                t_min.x = (V000_.x - p.x) / d.x;   //min
                t_max.x = (V111_.x - p.x) / d.x;   //max
                side.x  = mqi::XM;
            } else {                               //if intersect, X-
                t_max.x = (V000_.x - p.x) / d.x;   //max
                t_min.x = (V111_.x - p.x) / d.x;   //min
                side.x  = mqi::XP;
            }

        } else {
            // direction is near zero
            d.x     = 0;
            t_min.x = mqi_m_inf;
            t_max.x = mqi::p_inf;
        }   //----< X

        ///< check Y axis
        me = d.dot(n010_);
        pe = p.dot(n010_);
        if (me * me > mqi::near_zero) {
            if (me > 0) {                          //if intersect, Y+
                t_min.y = (V000_.y - p.y) / d.y;   //min
                t_max.y = (V111_.y - p.y) / d.y;   //max
                side.y  = mqi::YM;
            } else {                               //if intersect, Y-
                t_max.y = (V000_.y - p.y) / d.y;   //max
                t_min.y = (V111_.y - p.y) / d.y;   //min
                side.y  = mqi::YP;
            }
        } else {
            d.y     = 0;
            t_min.y = mqi_m_inf;
            t_max.y = mqi::p_inf;
        }   //----< Y
        ///< check Z (3rd) axis
        me = d.dot(n001_);
        pe = p.dot(n001_);
        if (me * me > mqi::near_zero) {
            if (me > 0) {                          //if intersect, Z+
                t_min.z = (V000_.z - p.z) / d.z;   //min
                t_max.z = (V111_.z - p.z) / d.z;   //max
                side.z  = mqi::ZM;
            } else {                               //if intersect, Z-
                t_max.z = (V000_.z - p.z) / d.z;   //max
                t_min.z = (V111_.z - p.z) / d.z;   //min
                side.z  = mqi::ZP;
            }
        } else {
            d.z     = 0;
            t_min.z = mqi_m_inf;
            t_max.z = mqi::p_inf;
        }

        ///< Find max value among T_min of x,y,z axis
        R u_min;
        if (t_min.x > t_min.y) {
            u_min = (t_min.x > t_min.z) ? t_min.x : t_min.z;
        } else {
            u_min = (t_min.y > t_min.z) ? t_min.y : t_min.z;
        }   //u_min
        ///< Find min value among T_max of x,y,z axis
        R u_max;
        if (t_max.x < t_max.y) {
            u_max = (t_max.x < t_max.z) ? t_max.x : t_max.z;
        } else {
            u_max = (t_max.y < t_max.z) ? t_max.y : t_max.z;
        }
#ifdef DEBUG
        printf("t_max %.5f %.5f %.5f\n", t_max.x, t_max.y, t_max.z);
        printf("t_min %.5f %.5f %.5f\n", t_min.x, t_min.y, t_min.z);
        printf("u_min %.5f u_max %.5f dist %.5f\n", u_min, u_max, u_max - u_min);
#endif
        if ((u_min < u_max || std::abs(u_min - u_max) < mqi::geometry_tolerance) && u_min >= 0 &&
            u_max >= 0) {
            its.dist          = u_min;
            mqi::vec3<R> p_on = p + d * its.dist;
            its.cell          = this->index(p_on, d);
        } else {
            return its;
        }
        return its;
    }   //< intersect : ray from outside

    CUDA_HOST_DEVICE
    inline mqi::vec3<ijk_t>
    index(const mqi::vec3<R>& p, mqi::vec3<R>& dir)   // if p is on boundary
    {   //find index for the first intersection, return voxel index
        mqi::vec3<ijk_t> idx;
        R                min_x = 100.0, min_y = 100.0, min_z = 100.0;
        for (int ind = 0; ind < dim_.x; ind++) {
            if (mqi::mqi_abs(xe_[ind] - p.x) < mqi::geometry_tolerance) {
                if (dir.x > 0) {
                    idx.x = ind;
                    break;
                } else if (dir.x < 0) {
                    idx.x = ind - 1;
                    break;
                } else {
                    idx.x = ind;
                    break;
                }
            } else if (mqi::mqi_abs(xe_[ind + 1] - p.x) < mqi::geometry_tolerance) {
                if (dir.x > 0) {
                    idx.x = ind + 1;
                    break;
                } else if (dir.x < 0) {
                    idx.x = ind;
                    break;
                } else {
                    idx.x = ind;
                    break;
                }
            } else if (xe_[ind] - p.x < 0 && xe_[ind + 1] - p.x > 0) {
                idx.x = ind;
                break;
            } else {
                idx.x = -1;
            }
        }

        for (int ind = 0; ind < dim_.y; ind++) {
            if (mqi::mqi_abs(ye_[ind] - p.y) < mqi::geometry_tolerance) {
                if (dir.y > 0) {
                    idx.y = ind;
                    break;
                } else if (dir.y < 0) {
                    idx.y = ind - 1;
                    break;
                } else {
                    idx.y = ind;
                    break;
                }
            } else if (mqi::mqi_abs(ye_[ind + 1] - p.y) < mqi::geometry_tolerance) {
                if (dir.y > 0) {
                    idx.y = ind + 1;
                    break;
                } else if (dir.y < 0) {
                    idx.y = ind;
                    break;
                } else {
                    idx.y = ind;
                    break;
                }
            } else if (ye_[ind] - p.y < 0 && ye_[ind + 1] - p.y > 0) {
                idx.y = ind;
                break;
            } else {
                idx.y = -1;
            }
        }

        for (int ind = 0; ind < dim_.z; ind++) {
            if (mqi::mqi_abs(ze_[ind] - p.z) < mqi::geometry_tolerance) {
                if (dir.z > 0) {
                    idx.z = ind;
                    break;
                } else if (dir.z < 0) {
                    idx.z = ind - 1;
                    break;
                } else {
                    idx.z = ind;
                    break;
                }
            } else if (mqi::mqi_abs(ze_[ind + 1] - p.z) < mqi::geometry_tolerance) {
                if (dir.z > 0) {
                    idx.z = ind + 1;
                    break;
                } else if (dir.z < 0) {
                    idx.z = ind;
                    break;
                } else {
                    idx.z = ind;
                    break;
                }
            } else if (ze_[ind] - p.z < 0 && ze_[ind + 1] - p.z > 0) {
                idx.z = ind;
                break;
            } else {
                idx.z = -1;
            }
        }
        return idx;
    }

    CUDA_HOST_DEVICE
    inline void
    index(mqi::vec3<R>& vtx1, mqi::vec3<R>& dir1, mqi::vec3<ijk_t>& idx) {
        //find intersection for the rest of the intersection, retun voxel index
        /// Do we need to deal with exceptions? Like, p.x-xe_[idx.x]<dx which must not happend

        int flag_x = 0, flag_y = 0, flag_z = 0;
        R   x_edge1, x_edge2;
        if (dir1.x < 0 &&
            (mqi::mqi_abs(vtx1.x - xe_[idx.x]) < mqi::geometry_tolerance || vtx1.x < xe_[idx.x])) {
            idx.x -= 1;
        } else if (dir1.x > 0 && (mqi::mqi_abs(vtx1.x - xe_[idx.x + 1]) < mqi::geometry_tolerance ||
                                  vtx1.x > xe_[idx.x + 1])) {
            idx.x += 1;
        }

        if (dir1.y < 0 &&
            (mqi::mqi_abs(vtx1.y - ye_[idx.y]) < mqi::geometry_tolerance || vtx1.y < ye_[idx.y])) {
            idx.y -= 1;
        } else if (dir1.y > 0 && (mqi::mqi_abs(vtx1.y - ye_[idx.y + 1]) < mqi::geometry_tolerance ||
                                  vtx1.y > ye_[idx.y + 1])) {
            idx.y += 1;
        }

        if (dir1.z < 0 &&
            (mqi::mqi_abs(vtx1.z - ze_[idx.z]) < mqi::geometry_tolerance || vtx1.z < ze_[idx.z])) {
            idx.z -= 1;
        } else if (dir1.z > 0 && (mqi::mqi_abs(vtx1.z - ze_[idx.z + 1]) < mqi::geometry_tolerance ||
                                  vtx1.z > ze_[idx.z + 1])) {
            idx.z += 1;
        }
    }

    ///< check the index is valid.
    ///< this can be used to determine whether next index is out-of-grid
    CUDA_HOST_DEVICE
    inline bool
    is_valid(mqi::vec3<ijk_t>& c) {
        if (c.x < 0 || c.y < 0 || c.z < 0) return false;
        if (c.x >= dim_.x || c.y >= dim_.y || c.z >= dim_.z) return false;
        //            if (c.x > dim_.x || c.y > dim_.y || c.z > dim_.z) return false;

        return true;
    }
};

}   // namespace mqi

#endif
