#ifndef MQI_SCORER_ENERGY_DEPOSIT_HPP
#define MQI_SCORER_ENERGY_DEPOSIT_HPP

#include <moqui/base/mqi_grid3d.hpp>
#include <moqui/base/mqi_material.hpp>
#include <moqui/base/mqi_track.hpp>

namespace mqi
{

///< Basic function to scorer all energy deposit
template<typename R>
CUDA_DEVICE double
energy_deposit(const track_t<R>& trk, const cnb_t& cnb, grid3d<mqi::density_t, R>& geo) {
    //    printf("energy deposit %f %f\n",trk.dE,trk.dE*trk.dE);
    return trk.dE + trk.local_dE;
}

///< Function to scorer energy deposit by primary only
template<typename R>
CUDA_DEVICE double
energy_deposit_primary(const track_t<R>& trk, const cnb_t& cnb, grid3d<mqi::density_t, R>& geo) {
    return trk.primary == true ? trk.dE : 0.0;
}

///< Function to scorer energy deposit by secondary
template<typename R>
CUDA_DEVICE double
energy_deposit_secondary(const track_t<R>& trk, const cnb_t& cnb, grid3d<mqi::density_t, R>& geo) {
    return trk.primary == false ? trk.dE : 0.0;
}

template<typename R>
CUDA_DEVICE double
dose_to_water(const track_t<R>& trk, const cnb_t& cnb, grid3d<mqi::density_t, R>& geo) {
    R density;
#if defined(__CUDACC__)
    //    density = __half2float(geo.get_data()[cnb]);
    density = geo.get_data()[cnb];
#else
    density = geo.get_data()[cnb];
#endif
    R             volume = geo.get_volume(cnb);
    mqi::h2o_t<R> water;
    if (density < 1.0e-7) {
        return 0.0;
    } else {
        water.rho_mass = density;
        return (trk.dE + trk.local_dE) * 1.60218e-10 /
               (volume * density * water.stopping_power_ratio(trk.vtx0.ke));
    }
}

template<typename R>
CUDA_DEVICE double
dose_to_medium(const track_t<R>& trk, const cnb_t& cnb, grid3d<mqi::density_t, R>& geo) {
    R density;
    density  = geo.get_data()[cnb];
    R volume = geo.get_volume(cnb);
    return trk.primary ? trk.dE * 1.60218e-13 * 1000.0 / (volume * density)
                       : 0.0;   // Convert to J/kg
}

template<typename R>
CUDA_DEVICE double
LETd_weight1(const track_t<R>& trk, const cnb_t& cnb, grid3d<mqi::density_t, R>& geo) {
    R density;
    density = geo.get_data()[cnb];
    density *= 1000.0;
    double length = (trk.vtx1.pos.x - trk.vtx0.pos.x) * (trk.vtx1.pos.x - trk.vtx0.pos.x);
    length += (trk.vtx1.pos.y - trk.vtx0.pos.y) * (trk.vtx1.pos.y - trk.vtx0.pos.y);
    length += (trk.vtx1.pos.z - trk.vtx0.pos.z) * (trk.vtx1.pos.z - trk.vtx0.pos.z);
    length = mqi::mqi_sqrt(length);
    if (length <= 0) { return 0.0; }
    double let = trk.dE / length / density;
    if (let >= 25.0) {
        return 0;
    } else {
        return trk.dE * let;
    }
}

template<typename R>
CUDA_DEVICE double
LETd_weight2(const track_t<R>& trk, const cnb_t& cnb, grid3d<mqi::density_t, R>& geo) {
    R density;
    density = geo.get_data()[cnb];
    density *= 1000.0;
    double length = (trk.vtx1.pos.x - trk.vtx0.pos.x) * (trk.vtx1.pos.x - trk.vtx0.pos.x);
    length += (trk.vtx1.pos.y - trk.vtx0.pos.y) * (trk.vtx1.pos.y - trk.vtx0.pos.y);
    length += (trk.vtx1.pos.z - trk.vtx0.pos.z) * (trk.vtx1.pos.z - trk.vtx0.pos.z);
    length = mqi::mqi_sqrt(length);
    if (length <= 0) { return 0.0; }
    double let = trk.dE / length / density;
    if (let >= 25.0) {
        return 0;
    } else {
        return trk.dE * 1.0;
    }
}

template<typename R>
CUDA_DEVICE double
LETt_weight1(const track_t<R>& trk, const cnb_t& cnb, grid3d<mqi::density_t, R>& geo) {
    R density;
    density = geo.get_data()[cnb];
    density *= 1000.0;
    double length = (trk.vtx1.pos.x - trk.vtx0.pos.x) * (trk.vtx1.pos.x - trk.vtx0.pos.x);
    length += (trk.vtx1.pos.y - trk.vtx0.pos.y) * (trk.vtx1.pos.y - trk.vtx0.pos.y);
    length += (trk.vtx1.pos.z - trk.vtx0.pos.z) * (trk.vtx1.pos.z - trk.vtx0.pos.z);
    length = mqi::mqi_sqrt(length);
    if (length <= 0) { return 0.0; }
    double let = trk.dE / length / density;
    return length * let;
}

template<typename R>
CUDA_DEVICE double
LETt_weight2(const track_t<R>& trk, const cnb_t& cnb, grid3d<mqi::density_t, R>& geo) {
    R density;
    density = geo.get_data()[cnb];
    density *= 1000.0;
    double length = (trk.vtx1.pos.x - trk.vtx0.pos.x) * (trk.vtx1.pos.x - trk.vtx0.pos.x);
    length += (trk.vtx1.pos.y - trk.vtx0.pos.y) * (trk.vtx1.pos.y - trk.vtx0.pos.y);
    length += (trk.vtx1.pos.z - trk.vtx0.pos.z) * (trk.vtx1.pos.z - trk.vtx0.pos.z);
    length = mqi::mqi_sqrt(length);
    //    if (length < 0 || mqi::mqi_abs(length) < 1e-3) { return 0.0; }
    if (length <= 0) { return 0.0; }
    return length;
}

#if defined(__CUDACC__)
CUDA_DEVICE fp_compute_hit<mqi::phsp_t> energy_deposit_pointer = mqi::energy_deposit;
CUDA_DEVICE fp_compute_hit<mqi::phsp_t> energy_deposit_primary_pointer =
  mqi::energy_deposit_primary;
CUDA_DEVICE fp_compute_hit<mqi::phsp_t> Dm_pointer           = mqi::dose_to_medium;
CUDA_DEVICE fp_compute_hit<mqi::phsp_t> Dw_pointer           = mqi::dose_to_water;
CUDA_DEVICE fp_compute_hit<mqi::phsp_t> LETd_weight1_pointer = mqi::LETd_weight1;
CUDA_DEVICE fp_compute_hit<mqi::phsp_t> LETd_weight2_pointer = mqi::LETd_weight2;
CUDA_DEVICE fp_compute_hit<mqi::phsp_t> LETt_weight1_pointer = mqi::LETt_weight1;
CUDA_DEVICE fp_compute_hit<mqi::phsp_t> LETt_weight2_pointer = mqi::LETt_weight2;

#endif
}   // namespace mqi
#endif
