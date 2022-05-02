#ifndef MQI_MATERIAL_HPP
#define MQI_MATERIAL_HPP

#include <moqui/base/mqi_common.hpp>
#include <moqui/base/mqi_math.hpp>
#include <moqui/base/mqi_physics_constants.hpp>

namespace mqi
{

typedef uint16_t material_id;
///< Interaction model (pure virtual class)
///< interface between particle and material
///< use template R for density type
///<
template<typename R>
class material_t
{
public:
    physics_constants<R> units;   //TODO: better to be defined as global variable.

    R        two_pi_re2_mc2_nel;   //two_pi_re2_mc2_h2o * rho_mass
    R        rho_mass;             //g/mm^3
    R        rho_elec;             //electron density
    R        Z;                    //Atomic number
    R        weight;               //molecular weight (g)
    R        electrons;            //number of electrons
    R        Iev;                  //ionization potential
    R        Iev_sq;               //Iev * Iev for stopping power calculation
    R        X0;                   //Radiation length
    uint16_t id;

public:
    CUDA_HOST_DEVICE
    material_t() {
        ;
    }

    CUDA_HOST_DEVICE
    ~material_t() {
        ;
    }

    CUDA_HOST_DEVICE
    material_t<R>&
    operator=(const material_t<R>& r) {
        two_pi_re2_mc2_nel = r.two_pi_re2_mc2_nel;
        rho_mass           = r.rho_mass;
        rho_elec           = r.rho_elec;
        Z                  = r.Z;
        weight             = r.weight;
        electrons          = r.electrons;
        Iev                = r.Iev;
        X0                 = r.X0;
        return *this;
    }

    ///< variable density
    CUDA_HOST_DEVICE
    inline virtual R
    mass_density(R scale = 1.0) const {
        return scale * rho_mass;
    }

    CUDA_HOST_DEVICE
    inline virtual R
    dedx_term0() const {
        return two_pi_re2_mc2_nel;
    }

    CUDA_HOST_DEVICE
    inline virtual R
    atomic_number() const {
        return Z;
    }

    ///< variable density
    //    CUDA_HOST_DEVICE
    CUDA_DEVICE
    inline virtual R
    stopping_power_ratio(R Ek, int8_t id = -1) {
        ////< 0.9 g/cm^3 ->  g/mm^3
        R density_tmp = this->rho_mass * 1000.0;
        //// Fippel
        if (density_tmp <= 0.26) {
            if (density_tmp < 0.0012) {
                return 0.0;
            } else {
                return mqi::intpl1d<R>(density_tmp, 0.0012, 0.26, 0.8815, 0.9925);
            }
        } else {
            ///< Ek : proton kinetic energy
            R rsp = 1.0123 - 3.386e-5 * Ek;
            rsp += 0.291 * (1.0 + mqi::mqi_pow(Ek, static_cast<R>(-0.3421))) *
                   (mqi::mqi_pow(density_tmp, static_cast<R>(-0.7)) - 1.0);
            if (density_tmp >= 0.9) {
                return rsp;
            } else {
                return mqi::intpl1d<R>(density_tmp, 0.26, 0.9, 0.9925, rsp);
            }
        }
    }
    ///< variable density
    CUDA_HOST_DEVICE
    virtual R
    radiation_length() {
        R radiation_length_mat = 0.0;
        R f                    = 0.0;
        R density_tmp          = this->rho_mass * 1000.0;
        if (density_tmp <= 0.26) {
            f = 0.9857 + 0.0085 * density_tmp;
        } else if (density_tmp > 0.26 && density_tmp <= 0.9) {
            f = 1.0446 - 0.2180 * density_tmp;
        } else if (density_tmp > 0.9) {
            //            f = 1.19 + 0.44 * mqi::mqi_ln(density_tmp - 0.44);
            f = 1.19 + 0.44 * logf(density_tmp - 0.44);
        }
        radiation_length_mat = (this->units.water_density * this->units.radiation_length_water) /
                               (density_tmp * 1e-3 * f);
        return radiation_length_mat;
    }
};

///< water_t
template<typename R>
class h2o_t : public material_t<R>
{
public:
    CUDA_HOST_DEVICE
    h2o_t() : material_t<R>() {
        this->rho_mass           = 1.0 / material_t<R>::units.cm3;          //0.001 mm^3
        this->rho_elec           = 3.3428e+23 / material_t<R>::units.cm3;   //mm^-3
        this->two_pi_re2_mc2_nel = material_t<R>::units.two_pi_re2_mc2_h2o;
        this->Iev                = 78.0;   //eV;
        this->Iev_sq             = 78.0 * 78.0;
        this->Z                  = 18;
        this->X0                 = 36.0863 * this->units.cm;
    }

    CUDA_HOST_DEVICE
    ~h2o_t() {
        ;
    }
};

///< water_t
template<typename R>
class air_t : public material_t<R>
{
public:
    CUDA_HOST_DEVICE
    air_t() : material_t<R>() {
        ///< TODO: fill the values for air
        this->rho_mass           = 0.0012047 / material_t<R>::units.cm3;    ///checked with NIST
        this->rho_elec           = 3.3428e+23 / material_t<R>::units.cm3;   /// mm cm^(-3)
        this->two_pi_re2_mc2_nel = material_t<R>::units.two_pi_re2_mc2_h2o * this->rho_mass;
        this->Iev                = 85.7;
        this->Iev_sq             = 85.7 * 85.7;
        this->Z                  = 18;
        this->X0 = (36.62 * this->units.cm) * this->rho_mass;   //36.0863/this->units.cm;
    }

    CUDA_HOST_DEVICE
    ~air_t() {
        ;
    }
};

}   // namespace mqi

#endif
