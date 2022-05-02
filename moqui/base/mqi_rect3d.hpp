#ifndef MQI_RECT3D_H
#define MQI_RECT3D_H

/// \file
///
/// rectlinear geometry to represent CT, doe, DVF, etc.

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <valarray>

#include <moqui/base/mqi_common.hpp>
#include <moqui/base/mqi_vec.hpp>

namespace mqi
{

/// \class rect3d
/// \tparam T for grid values, e.g., dose, HU, vector
/// \tparam R for grid coordinates, float, double, etc.
template<typename T, typename R>
class rect3d
{

public:
    /// Enumerate for a corner position
    typedef enum
    {
        //cXYZ, 0 is less, 1 is greater
        c000 = 0,
        c100,
        c110,
        c010,
        c001,
        c101,
        c111,
        c011,
        cXXX
    } cell_corner;

protected:
    /// the center of pixel
    R* x_;   ///< x axis values.
    R* y_;   ///< y axis values.
    R* z_;   ///< z axis values.

    //To make navigation and interpolation problems simple, we assume Transport matrix
    //X: 1, 0, 0 Y: 0, 1, 0 Z: 0, 0, 1
    //or set one of following flag
    bool flip_[3] = { false, false, false };

    //Edge data is not neccessary for 4D interpolation but important for navigation
    //R  x_edge[2]; /**< x boundary of entire rectlinear, e.g., x_edge[0] for x min, x_edge[1] for x max*/
    //R  y_edge[2]; /**< y boundary of entire rectlinear, e.g., y_edge[0] for y min, y_edge[1] for y max*/
    //R  z_edge[2]; /**< y boundary of entire rectlinear, e.g., z_edge[0] for z min, z_edge[1] for z max*/

    mqi::vec3<ijk_t> dim_;   ///< Number of voxels: dim_.x, dim_.y, dim_.z

    std::valarray<T> data_;   ///< Data in this rectlinear

public:
    /// Default constructor only for child classes
    ///cuda_host_device or cuda_host
    rect3d() {
        ;
    }

    /// Construct a rectlinear grid from vectors of x/y/z
    /// \param x,y,z  1D vector of central points of voxels along x-axis
    /// for example, -1,0,1 mean three voxels along x.
    CUDA_HOST_DEVICE
    rect3d(std::vector<R>& x, std::vector<R>& y, std::vector<R>& z) {
        x_ = new R[x.size()];
        y_ = new R[y.size()];
        z_ = new R[z.size()];

        for (size_t i = 0; i < x.size(); ++i)
            x_[i] = x[i];
        for (size_t i = 0; i < y.size(); ++i)
            y_[i] = y[i];
        for (size_t i = 0; i < z.size(); ++i)
            z_[i] = z[i];

        dim_.x = x.size();
        dim_.y = y.size();
        dim_.z = z.size();
    }

    /// Construct a rectlinear grid from array of x/y/z with their size
    /// \param x,y,z  1D array of central points of voxels along x-axis
    /// \param xn,yn,zn  size of 1D array for points.
    CUDA_HOST_DEVICE
    rect3d(R x[], int xn, R y[], int yn, R z[], int zn) {
        x_ = new R[xn];
        y_ = new R[yn];
        z_ = new R[zn];

        for (size_t i = 0; i < xn; ++i)
            x_[i] = x[i];
        for (size_t i = 0; i < yn; ++i)
            y_[i] = y[i];
        for (size_t i = 0; i < zn; ++i)
            z_[i] = z[i];

        dim_.x = xn;
        dim_.y = yn;
        dim_.z = zn;
    }

    /// Copy constructor
    CUDA_HOST_DEVICE
    rect3d(rect3d& c) {
        dim_ = c.dim_;

        x_ = new R[dim_.x];
        y_ = new R[dim_.y];
        z_ = new R[dim_.z];

        for (size_t i = 0; i < dim_.x; ++i)
            x_[i] = c.x_[i];
        for (size_t i = 0; i < dim_.y; ++i)
            y_[i] = c.y_[i];
        for (size_t i = 0; i < dim_.z; ++i)
            z_[i] = c.z_[i];
    }

    /// Destructor releases dynamic allocation for x/y/z coordinates
    CUDA_HOST_DEVICE
    ~rect3d() {
        delete x_;
        delete y_;
        delete z_;
    }

    /// Returns the interpolated value for given point, p. in std::array type
    /// \param p is a std::array to represent a position, p[0], p[1], p[2] for x, y, z.
    virtual T
    operator()(const std::array<R, 3> p) {
        return operator()(mqi::vec3<R>(p[0], p[1], p[2]));
    }

    /// Return the interpolated value for given point, x, y, z
    /// \param x, y, z are for position
    virtual T
    operator()(const R x, const R y, const R z) {
        return operator()(mqi::vec3<R>(x, y, z));
    }

    /// Returns data
    /// \return data_
    const std::valarray<T>&
    get_data() const {
        return data_;
    }

    /// Returns x center positions
    /// \return x_
    const R*
    get_x() const {
        return x_;
    }

    /// Returns y center positions
    /// \return y_
    const R*
    get_y() const {
        return y_;
    }

    /// Returns z center positions
    /// \return z_
    const R*
    get_z() const {
        return z_;
    }

    /// Returns the interpolated value for given point, x, y, z
    /// \param pos is a type of mqi::vec3<R>.
    virtual T
    operator()(const mqi::vec3<R>& pos) {
        std::array<size_t, 3>  c000_idx = this->find_c000_index(pos);
        const std::array<R, 6> cell_pts = this->cell_position(c000_idx);
        const std::array<T, 8> coner    = this->cell_data(c000_idx);

        R xd = (pos.x - cell_pts[0]) / (cell_pts[1] - cell_pts[0]);
        R yd = (pos.y - cell_pts[2]) / (cell_pts[3] - cell_pts[2]);
        R zd = (pos.z - cell_pts[4]) / (cell_pts[5] - cell_pts[4]);

        T c00 = coner[c000] * (1.0 - xd) + coner[c100] * xd;
        T c10 = coner[c010] * (1.0 - xd) + coner[c110] * xd;

        T c01 = coner[c001] * (1.0 - xd) + coner[c101] * xd;
        T c11 = coner[c011] * (1.0 - xd) + coner[c111] * xd;

        T c0 = c00 * (1.0 - yd) + c10 * yd;
        T c1 = c01 * (1.0 - yd) + c11 * yd;

        return c0 * (1.0 - zd) + c1 * zd;
    }

    /// Returns the data value for given x/y/z index
    /// \param p index, p[0], p[1], p[2] for x, y, z.
    virtual T
    operator[](const std::array<size_t, 3> p)   //&
    {
        return data_[ijk2data(p[0], p[1], p[2])];
    }

    /// Returns the data value for given x/y/z index
    /// \param[in] p index, p[0], p[1], p[2] for x, y, z.
    virtual T
    operator[](const std::array<int, 3> p)   //&
    {
        return data_[ijk2data(p[0], p[1], p[2])];
    }

    /// Returns a total of 8 coner data of a cell
    /// \param c000_idx an array of cell ids in x, y, and z.
    /// \return coner values of a cell, C000, C001, C100, etc.
    virtual std::array<T, 8>
    cell_data(const std::array<size_t, 3>& c000_idx) {
        size_t x0 = c000_idx[0];
        size_t x1 = x0 + 1;
        size_t y0 = c000_idx[1];
        size_t y1 = y0 + 1;
        size_t z0 = c000_idx[2];
        size_t z1 = z0 + 1;

        std::array<T, 8> coner = { data_[ijk2data(x0, y0, z0)], data_[ijk2data(x1, y0, z0)],
                                   data_[ijk2data(x1, y1, z0)], data_[ijk2data(x0, y1, z0)],
                                   data_[ijk2data(x0, y0, z1)], data_[ijk2data(x1, y0, z1)],
                                   data_[ijk2data(x1, y1, z1)], data_[ijk2data(x0, y1, z1)] };

        return coner;
    }

    /// Returns a total of 6 values of position
    /// \param c000_idx an array of cell index (x, y, and z).
    /// \return x0 and x1, y0 and y1, and z0 and z1.
    virtual std::array<R, 6>
    cell_position(const std::array<size_t, 3>& c000_idx) {

        size_t xpi = c000_idx[0];
        size_t ypi = c000_idx[1];
        size_t zpi = c000_idx[2];

        std::array<R, 6> pts;
        pts[0] = x_[xpi];
        pts[1] = x_[xpi + 1];
        pts[2] = y_[ypi];
        pts[3] = y_[ypi + 1];
        pts[4] = z_[zpi];
        pts[5] = z_[zpi + 1];

        return pts;
    }

    /// Returns a coner index (x,y,z) where a given position is placed within (x+1,y+1,z+1)
    /// \param pos cartesian coordinate of x, y, and z.
    /// \return x,y,z indices of a cell
    CUDA_HOST
    virtual std::array<size_t, 3>
    find_c000_index(const mqi::vec3<R>& pos) {
        std::array<size_t, 3> c000_idx;
        c000_idx[0] = this->find_c000_x_index(pos.x);
        c000_idx[1] = this->find_c000_y_index(pos.y);
        c000_idx[2] = this->find_c000_z_index(pos.z);

        return c000_idx;
    }

    /// Returns a coner index of x
    /// \param pos cartesian coordinate of x
    /// \return x index of a cell
    CUDA_HOST
    inline virtual size_t
    find_c000_x_index(const R& x) {
        //In case this code runs on GPU, consider to implement binary_search algorithm or use thrust
        //but thrust performance is not so good from
        //https://groups.google.com/forum/#!topic/thrust-users/kTX6lgntOAc
        R* i = std::lower_bound(x_, x_ + dim_.x, x, std::less_equal<R>());
        return i - x_ - 1;
    }

    /// Returns a coner index of y
    /// \param pos cartesian coordinate of y
    /// \return y index of a cell
    CUDA_HOST
    inline virtual size_t
    find_c000_y_index(const R& y) {
        R* j = std::lower_bound(y_, y_ + dim_.y, y, std::less_equal<R>());
        return j - y_ - 1;
    }

    /// Returns a coner index of z
    /// \param pos cartesian coordinate of z
    /// \return z index of a cell
    inline virtual size_t
    find_c000_z_index(const R& z) {
        R* k = std::lower_bound(z_, z_ + dim_.z, z, std::less_equal<R>());
        return k - z_ - 1;
    }

    /// Returns whether the point is in the grid or not
    /// \param pos cartesian coordinate of x,y,z
    /// \return true if the p is in grid or false
    CUDA_HOST_DEVICE
    inline virtual bool
    is_in_point(const vec3<R>& p) {
        ///< check p is inside of pixel grid not entire volume
        ///< Min/Max of src

        if (p.x < x_[0] || p.x >= x_[dim_.x - 1]) return false;
        if (p.y < y_[0] || p.y >= y_[dim_.y - 1]) return false;
        if (p.z < z_[0] || p.z >= z_[dim_.z - 1]) return false;

        return true;
    }

    /// Returns whether the point is in the rect including edge or not
    /// \param pos cartesian coordinate of x,y,z
    /// \return true if the p is in grid or false
    CUDA_HOST_DEVICE
    inline virtual bool
    is_in_rect(const vec3<R>& p) {
        //if( p.x < xedge_[0] || p.x >= xedge_[1]) return false;
        return true;
    }

    /// Returns the center position of a rect
    /// \note more accurately, the center should be middle of edge
    /// not middle of first and last point cause the thickness may be differents
    CUDA_HOST_DEVICE
    mqi::vec3<R>
    get_center() {
        return mqi::vec3<R>(0.5 * (x_[0] + x_[dim_.x - 1]),
                            0.5 * (y_[0] + y_[dim_.y - 1]),
                            0.5 * (z_[0] + z_[dim_.z - 1]));
    }

    /// Returns size of box
    /// note: distance between edges
    /// the firt/last pixel thickness is assumed to be
    /// a half of distance between first and second or last and before-last
    CUDA_HOST_DEVICE
    mqi::vec3<R>
    get_size() {
        R Lx = x_[dim_.x - 1] - x_[0];
        Lx += 0.5 * (x_[dim_.x - 1] - x_[dim_.x - 2]);
        Lx += 0.5 * (x_[1] - x_[0]);

        R Ly = y_[dim_.y - 1] - y_[0];
        Ly += 0.5 * (y_[dim_.y - 1] - y_[dim_.y - 2]);
        Ly += 0.5 * (y_[1] - y_[0]);

        R Lz = z_[dim_.z - 1] - z_[0];
        Lz += 0.5 * (z_[dim_.z - 1] - z_[dim_.z - 2]);
        Lz += 0.5 * (z_[1] - z_[0]);

        return mqi::vec3<R>(Lx, Ly, Lz);
    }

    /// Returns number of bins box
    CUDA_HOST_DEVICE
    mqi::vec3<ijk_t>
    get_nxyz() {
        return dim_;
    }

    CUDA_HOST_DEVICE
    void
    set_nxyz(mqi::vec3<ijk_t> dim) {
        dim_ = dim;
    }
    /// Returns position of the first corner cell
    CUDA_HOST_DEVICE
    mqi::vec3<R>
    get_origin() {
        return mqi::vec3<R>(x_[0], y_[0], z_[0]);
    }

    /// Prints out x,y,z coordinate positions
    CUDA_HOST
    virtual void
    dump_pts() {
        std::cout << "X: ";
        for (size_t i = 0; i < dim_.x; ++i) {
            std::cout << " " << x_[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "Y: ";
        for (size_t i = 0; i < dim_.y; ++i) {
            std::cout << " " << y_[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "Z: ";
        for (size_t i = 0; i < dim_.z; ++i) {
            std::cout << " " << z_[i] << " ";
        }
        std::cout << std::endl;
    }

    /// Converts index of x,y,z to index of valarray(data)
    CUDA_HOST_DEVICE
    virtual inline size_t
    ijk2data(size_t i, size_t j, size_t k) {
        return k * dim_.x * dim_.y + j * dim_.x + i;
    }

    /// Writes data into file
    CUDA_HOST
    virtual void
    write_data(const std::string filename) {
        std::ofstream file1(filename, std::ios::out | std::ofstream::binary);
        file1.write(reinterpret_cast<const char*>(&data_[0]), data_.size() * sizeof(T));
        file1.close();
    }

    /// Writes data in any type of valarray to a file
    template<class S>
    void
    write_data(std::valarray<S>& output, const std::string filename) {
        std::ofstream file1(filename, std::ios::out | std::ofstream::binary);
        file1.write(reinterpret_cast<const char*>(&output[0]), output.size() * sizeof(S));
        file1.close();
    }

    /*
    /// this is not yet implemented
    #include <vtkImageReader2.h>
    #include <vtkMetaImageWriter.h>
    #include <vtkSmartPointer.h>

    CUDA_HOST
    virtual void
    write_mha(const std::string filename){

        vtkSmartPointer<vtkMetaImageWriter> writer = vtkSmartPointer<vtkMetaImageWriter>::New();
        writer->SetInputConnection(reader->GetOutputPort());
        writer->SetInpt
        writer->SetCompression(false);
        writer->SetFileName( cl_opts["--output1"][0].c_str() );
        writer->Write();

        std::ofstream file1( filename, std::ios::out | std::ofstream::binary);
        file1.write(reinterpret_cast<const char *>(&data_[0]), data_.size() * sizeof(T));r
        file1.close();

    }
    */

    /// Initializes data, currently values are sum of index square for testing
    CUDA_HOST
    virtual void
    load_data() {
        data_.resize(dim_.x * dim_.y * dim_.z);
    }

    /// Reads data from other source,  currently values are sum of index square for testing
    /// //total == dim_.x*dim_.y*dim_.z
    /// //will copy
    /// in case src is CPU and dest GPU
    void
    read_data(T* src, size_t total) {
        data_(src, total);
    }

    void
    read_data(std::valarray<T> src) {
        data_ = src;
    }

    /// Fills data with a given value
    CUDA_HOST
    virtual void
    fill_data(T a) {
        data_.resize(dim_.x * dim_.y * dim_.z);
        data_ = a;
    }

    /// Checks data is flipped from x_, y_, z_ and flip the order of  coordinate
    /// This method should be called after constructor gets called
    CUDA_HOST
    void
    flip_xyz_if_any(void) {
        flip_[0] = (x_[1] < x_[0]) ? true : false;
        flip_[1] = (y_[1] < y_[0]) ? true : false;
        flip_[2] = (z_[1] < z_[0]) ? true : false;
        if (flip_[0]) std::reverse(x_, x_ + dim_.x);
        if (flip_[1]) std::reverse(y_, y_ + dim_.y);
        if (flip_[2]) std::reverse(z_, z_ + dim_.z);
    }

    /// Flip data with a given value
    CUDA_HOST
    void
    flip_data(void) {
        if (flip_[0] == false && flip_[1] == false && flip_[2] == false) { return; }

        std::valarray<T> tmp0(data_.size());   //temporal copy object
        tmp0             = data_;
        long int id_from = 0;
        long int id_to   = 0;

        long int idx = 0;
        long int idy = 0;
        long int idz = 0;

        for (int k = 0; k < dim_.z; ++k) {
            for (int j = 0; j < dim_.y; ++j) {
                for (int i = 0; i < dim_.x; ++i) {
                    idx = (flip_[0]) ? (dim_.x - 1 - i) : i;
                    idy = (flip_[1]) ? (dim_.y - 1 - j) * dim_.x : j * dim_.x;
                    idz = (flip_[2]) ? (dim_.z - 1 - k) * dim_.x * dim_.y : k * dim_.x * dim_.y;

                    id_to        = this->ijk2data(i, j, k);
                    id_from      = idz + idy + idx;
                    data_[id_to] = tmp0[id_from];
                }   //x
            }       //y
        }           //z
    }

    /// A friend function to copy grid information of src to dest
    template<typename T0, typename R0, typename T1, typename R1>
    friend void
    clone_structure(rect3d<T0, R0>& src, rect3d<T1, R1>& dest);

    /// A friend function to interpolate new rect3d from a source
    /// interpolate(ct, dose) : possible
    /// interpolate(dose, dose)
    /// interpolate(dvf, dose) : is not possible
    template<typename T0, typename R0, typename T1, typename R1>
    friend void
    interpolate(rect3d<T0, R0>& src, rect3d<T1, R1>& dest, T1& fill_value);

    /// A friend function to warp source data (rect3d) to destination (rect3d) using dvf.
    /// It will pull src data from ref thus DVF of ref->src is neccessary.
    /// In MIM, when we calculate DIR, ref should be chosen first.
    /// It is recommended that src, dest, dvf have all same resolution.
    template<typename T0, typename R0, typename S0>
    friend void
    warp_linear(rect3d<T0, R0>&            src,
                rect3d<T0, R0>&            ref,
                rect3d<mqi::vec3<S0>, R0>& dvf,
                T0                         fill_value);
};

/// R0 and R1 should be comparable
template<typename T0, typename R0, typename T1, typename R1>
void
clone_structure(rect3d<T0, R0>& src, rect3d<T1, R1>& dest) {
    dest.dim_ = src.dim_;

    dest.x_ = new R1[dest.dim_.x];
    dest.y_ = new R1[dest.dim_.y];
    dest.z_ = new R1[dest.dim_.z];

    for (size_t i = 0; i < dest.dim_.x; ++i)
        dest.x_[i] = src.x_[i];
    for (size_t i = 0; i < dest.dim_.y; ++i)
        dest.y_[i] = src.y_[i];
    for (size_t i = 0; i < dest.dim_.z; ++i)
        dest.z_[i] = src.z_[i];
}

/// Interpolates src and fill dest
/// R0 and R1 should be comparable
template<typename T0, typename R0, typename T1, typename R1>
void
interpolate(rect3d<T0, R0>& src, rect3d<T1, R1>& dest, T1& fill_value) {

    std::cout << src.x_[0] << ", " << src.y_[0] << ", " << src.z_[0] << std::endl;
    std::cout << dest.x_[0] << ", " << dest.y_[0] << ", " << dest.z_[0] << std::endl;

    ///< Number of voxels: Nx, Ny, Nz
    const size_t nX = dest.dim_.x;
    const size_t nY = dest.dim_.y;
    const size_t nZ = dest.dim_.z;

    ///< center point of destination
    dest.data_.resize(nX * nY * nZ);

    mqi::vec3<R1> p       = { dest.x_[0], dest.y_[0], dest.z_[0] };
    size_t        counter = 0;
    for (size_t k = 0; k < nZ; ++k) {
        p.z = dest.z_[k];
        for (size_t j = 0; j < nY; ++j) {
            p.y = dest.y_[j];
            for (size_t i = 0; i < nX; ++i) {
                p.x                                = dest.x_[i];
                dest.data_[dest.ijk2data(i, j, k)] = src.is_in_point(p) ? src(p) : fill_value;

            }   //x
        }       //y
    }           //z
}

/// Warping data in src using vector field
template<typename T0, typename R0, typename S0>
void
warp_linear(rect3d<T0, R0>&            src,
            rect3d<T0, R0>&            dest,
            rect3d<mqi::vec3<S0>, R0>& vf,
            T0                         fill_value) {
    ///< Number of voxels: Nx, Ny, Nz
    const size_t nX = dest.dim_.x;
    const size_t nY = dest.dim_.y;
    const size_t nZ = dest.dim_.z;

    dest.data_.resize(nX * nY * nZ);

    ///< Looping reference
    mqi::vec3<R0> p_dest(dest.x_[0], dest.y_[0], dest.z_[0]);

    for (size_t k = 0; k < nZ; ++k) {
        p_dest.z = dest.z_[k];
        for (size_t j = 0; j < nY; ++j) {
            p_dest.y = dest.y_[j];
            for (size_t i = 0; i < nX; ++i) {
                p_dest.x = dest.x_[i];

                T0 value;
                if (vf.is_in_point(p_dest)) {
                    ///
                    /// If destination point is in DVF grid points,
                    /// apply translation and then check new position is in source
                    /// Then, assign value by interpolating values at 8 coners surrounding new position.
                    ///
                    mqi::vec3<R0> p_new = p_dest + vf(p_dest);
                    value               = src.is_in_point(p_new) ? src(p_new) : fill_value;
                } else {
                    value = src.is_in_point(p_dest) ? src(p_dest) : fill_value;

                }   //vf.is_in_point
                dest.data_[dest.ijk2data(i, j, k)] = value;

            }   //x
        }       //y
    }           //z
}

}   // namespace mqi

#endif
