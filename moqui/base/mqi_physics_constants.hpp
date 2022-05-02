#ifndef MQI_PHYSICS_CONSTANTS_HPP
#define MQI_PHYSICS_CONSTANTS_HPP

#include <moqui/base/mqi_common.hpp>

namespace mqi
{

/*
	Water: RadL = 36.093 cm, Nucl.Int.Lengh = 75.375 cm
	electromagnetic coupling = 1.43996e-12 MeV*mm/(eplus^2) , eplus  =1 
	static const double classic_electr_radius  = elm_coupling/electron_mass_c2;
	*/
template<typename R>
struct physics_constants {
    const R mm                     = 1.0;   //default length unit
    const R cm                     = 10.0;
    const R cm3                    = cm * cm * cm;
    const R mm3                    = mm * mm * mm;
    const R MeV                    = 1.0;
    const R eV                     = 1e-6;
    const R Mp                     = 938.272046 * MeV;   // Mp/eV = proton mass in eV
    const R Mp_sq                  = Mp * Mp;
    const R Me                     = 0.510998928 * MeV;
    const R Mo                     = 14903.3460795634 * MeV;
    const R Mo_sq                  = Mo * Mo;
    const R MoMp                   = Mo / Mp;
    const R MoMp_sq                = MoMp * MoMp;
    const R MeMp                   = Me / Mp;                 // Me/Mp
    const R MeMp_sq                = MeMp * MeMp;             // (Me/Mp)^2
    const R re                     = 2.8179403262e-12 * mm;   //classic_electron_radius
    const R re_sq                  = re * re;
    const R two_pi_re2_mc2         = 2.0 * M_PI * re_sq * Me;
    const R two_pi_re2_mc2_h2o     = two_pi_re2_mc2 * 3.3428e+23 / cm3;
    const R water_density          = 1.0 / cm3;      //1g/mm3
    const R radiation_length_water = 36.0863 * cm;   //mm 360.863
    const R mev_to_joule           = 1.60218e-13;    // J/MeV
};

}   // namespace mqi
#endif
