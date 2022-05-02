#ifndef MQI_PO_INELASTIC_HPP
#define MQI_PO_INELASTIC_HPP

#include <moqui/base/mqi_interaction.hpp>
//#include <cassert>
namespace mqi
{

///< Cross-section from Geant4 Hard00, 0.1 MeV to 299.6 MeV with 0.5 MeV step
///< cm^2/g
CUDA_CONSTANT const float cs_po_i_g4_table[600] = {
    0.00000, 0.00000, 0.00096, 0.00460, 0.01201, 0.02453, 0.04343, 0.06979, 0.10449, 0.14818,
    0.20101, 0.26284, 0.33306, 0.41085, 0.49449, 0.58315, 0.67516, 0.76817, 0.86151, 0.95352,
    1.04285, 1.12850, 1.21013, 1.28641, 1.35768, 1.42358, 1.48414, 1.53901, 1.58083, 1.62265,
    1.64373, 1.66447, 1.68555, 1.70629, 1.72035, 1.73406, 1.74812, 1.76217, 1.77588, 1.78994,
    1.79161, 1.79328, 1.79495, 1.79663, 1.79395, 1.79094, 1.78826, 1.78559, 1.78258, 1.77990,
    1.76986, 1.75983, 1.74979, 1.73975, 1.72871, 1.71734, 1.70629, 1.69525, 1.68388, 1.67284,
    1.65946, 1.64607, 1.63269, 1.61931, 1.60592, 1.59254, 1.57916, 1.56578, 1.55239, 1.53901,
    1.52964, 1.52028, 1.51091, 1.50154, 1.49217, 1.48280, 1.47344, 1.46407, 1.45470, 1.44533,
    1.43630, 1.42727, 1.41823, 1.40920, 1.40017, 1.39113, 1.38210, 1.37307, 1.36403, 1.35500,
    1.34998, 1.34496, 1.33994, 1.33492, 1.32991, 1.32489, 1.31987, 1.31485, 1.30983, 1.30481,
    1.29812, 1.29143, 1.28474, 1.27805, 1.27136, 1.26467, 1.25797, 1.25128, 1.24459, 1.23790,
    1.23121, 1.22452, 1.21783, 1.21113, 1.20444, 1.19775, 1.19106, 1.18437, 1.17768, 1.17099,
    1.16597, 1.16095, 1.15593, 1.15091, 1.14589, 1.14088, 1.13586, 1.13084, 1.12582, 1.12080,
    1.11578, 1.11076, 1.10575, 1.10073, 1.09571, 1.09069, 1.08567, 1.08065, 1.07563, 1.07062,
    1.06894, 1.06727, 1.06560, 1.06392, 1.06225, 1.06058, 1.05891, 1.05723, 1.05556, 1.05389,
    1.05221, 1.05054, 1.04887, 1.04720, 1.04552, 1.04385, 1.04218, 1.04051, 1.03883, 1.03716,
    1.03616, 1.03515, 1.03415, 1.03314, 1.03214, 1.03114, 1.03013, 1.02913, 1.02813, 1.02712,
    1.02612, 1.02512, 1.02411, 1.02311, 1.02210, 1.02110, 1.02010, 1.01909, 1.01809, 1.01709,
    1.01508, 1.01341, 1.01140, 1.00972, 1.00772, 1.00604, 1.00404, 1.00236, 1.00036, 0.99868,
    0.99668, 0.99500, 0.99300, 0.99132, 0.98932, 0.98764, 0.98564, 0.98396, 0.98196, 0.98028,
    0.97961, 0.97928, 0.97861, 0.97828, 0.97761, 0.97727, 0.97660, 0.97627, 0.97560, 0.97526,
    0.97460, 0.97426, 0.97359, 0.97326, 0.97259, 0.97225, 0.97158, 0.97125, 0.97058, 0.97025,
    0.96958, 0.96924, 0.96857, 0.96824, 0.96757, 0.96723, 0.96657, 0.96623, 0.96556, 0.96523,
    0.96456, 0.96422, 0.96355, 0.96322, 0.96255, 0.96222, 0.96155, 0.96121, 0.96054, 0.96021,
    0.95987, 0.95954, 0.95921, 0.95887, 0.95854, 0.95820, 0.95787, 0.95753, 0.95720, 0.95686,
    0.95653, 0.95619, 0.95586, 0.95552, 0.95519, 0.95486, 0.95452, 0.95419, 0.95385, 0.95352,
    0.95318, 0.95285, 0.95251, 0.95218, 0.95184, 0.95151, 0.95118, 0.95084, 0.95051, 0.95017,
    0.94984, 0.94950, 0.94917, 0.94883, 0.94850, 0.94816, 0.94783, 0.94750, 0.94716, 0.94683,
    0.94616, 0.94549, 0.94482, 0.94415, 0.94348, 0.94281, 0.94214, 0.94147, 0.94080, 0.94013,
    0.93947, 0.93880, 0.93813, 0.93746, 0.93679, 0.93612, 0.93545, 0.93478, 0.93411, 0.93344,
    0.93344, 0.93344, 0.93344, 0.93344, 0.93344, 0.93344, 0.93344, 0.93344, 0.93344, 0.93344,
    0.93344, 0.93344, 0.93344, 0.93344, 0.93344, 0.93344, 0.93344, 0.93344, 0.93344, 0.93344,
    0.93344, 0.93311, 0.93311, 0.93311, 0.93311, 0.93277, 0.93277, 0.93277, 0.93277, 0.93244,
    0.93244, 0.93244, 0.93244, 0.93211, 0.93211, 0.93211, 0.93211, 0.93177, 0.93177, 0.93177,
    0.93177, 0.93144, 0.93144, 0.93144, 0.93144, 0.93110, 0.93110, 0.93110, 0.93110, 0.93077,
    0.93077, 0.93077, 0.93077, 0.93043, 0.93043, 0.93043, 0.93043, 0.93010, 0.93010, 0.93010,
    0.92976, 0.92976, 0.92943, 0.92943, 0.92909, 0.92909, 0.92876, 0.92876, 0.92842, 0.92842,
    0.92809, 0.92809, 0.92776, 0.92776, 0.92742, 0.92742, 0.92709, 0.92709, 0.92675, 0.92675,
    0.92642, 0.92642, 0.92608, 0.92608, 0.92575, 0.92575, 0.92541, 0.92541, 0.92508, 0.92508,
    0.92474, 0.92474, 0.92441, 0.92441, 0.92408, 0.92408, 0.92374, 0.92374, 0.92341, 0.92341,
    0.92307, 0.92307, 0.92274, 0.92274, 0.92240, 0.92207, 0.92207, 0.92173, 0.92173, 0.92140,
    0.92106, 0.92106, 0.92073, 0.92073, 0.92040, 0.92006, 0.92006, 0.91973, 0.91973, 0.91939,
    0.91906, 0.91906, 0.91872, 0.91872, 0.91839, 0.91805, 0.91805, 0.91772, 0.91772, 0.91738,
    0.91705, 0.91705, 0.91672, 0.91672, 0.91638, 0.91605, 0.91605, 0.91571, 0.91571, 0.91538,
    0.91504, 0.91504, 0.91471, 0.91471, 0.91437, 0.91404, 0.91404, 0.91370, 0.91370, 0.91337,
    0.91303, 0.91303, 0.91270, 0.91270, 0.91237, 0.91203, 0.91203, 0.91170, 0.91170, 0.91136,
    0.91103, 0.91103, 0.91069, 0.91069, 0.91036, 0.91002, 0.91002, 0.90969, 0.90969, 0.90935,
    0.90902, 0.90902, 0.90869, 0.90869, 0.90835, 0.90802, 0.90802, 0.90768, 0.90768, 0.90735,
    0.90701, 0.90701, 0.90668, 0.90668, 0.90634, 0.90601, 0.90601, 0.90567, 0.90567, 0.90534,
    0.90501, 0.90501, 0.90467, 0.90467, 0.90434, 0.90400, 0.90400, 0.90367, 0.90367, 0.90333,
    0.90333, 0.90333, 0.90333, 0.90333, 0.90367, 0.90367, 0.90367, 0.90367, 0.90367, 0.90367,
    0.90367, 0.90367, 0.90367, 0.90367, 0.90400, 0.90400, 0.90400, 0.90400, 0.90400, 0.90400,
    0.90400, 0.90400, 0.90400, 0.90400, 0.90434, 0.90434, 0.90434, 0.90434, 0.90434, 0.90434,
    0.90434, 0.90434, 0.90434, 0.90434, 0.90467, 0.90467, 0.90467, 0.90467, 0.90467, 0.90467,
    0.90467, 0.90467, 0.90467, 0.90467, 0.90501, 0.90501, 0.90501, 0.90501, 0.90501, 0.90501,
    0.90501, 0.90501, 0.90501, 0.90501, 0.90534, 0.90534, 0.90534, 0.90534, 0.90534, 0.90534,
    0.90534, 0.90534, 0.90534, 0.90534, 0.90567, 0.90567, 0.90567, 0.90567, 0.90567, 0.90567,
    0.90567, 0.90567, 0.90567, 0.90567, 0.90601, 0.90601, 0.90601, 0.90601, 0.90601, 0.90601,
    0.90601, 0.90601, 0.90601, 0.90601, 0.90634, 0.90634, 0.90634, 0.90634, 0.90634, 0.90634,
    0.90634, 0.90634, 0.90634, 0.90634, 0.90668, 0.90668, 0.90668, 0.90668, 0.90668, 0.90668
};
///< Proton-Oxygen inelastic interaction
template<typename R>
class po_inelastic : public interaction<R, mqi::PROTON>
{
public:
    R E_bind;   ///MeV, binding energy, const R E_bind doesn't work with CUDA_SHARED. class heirachy
    R E_mini;   ///MeV, minimum energy for PO inelastic

public:
    CUDA_HOST_DEVICE
    po_inelastic() {
        E_bind = 5.0;
        E_mini = 3.0;
    }

    CUDA_HOST_DEVICE
    ~po_inelastic() {
        ;
    }

    CUDA_HOST_DEVICE
    virtual R
    cross_section(const relativistic_quantities<R>& rel, const material_t<R>& mat) {
        R cs = 0;
        if (rel.Ek > 7 && rel.Ek < 250) {
            cs = 1.64 * (rel.Ek - 7.9);
            cs *= mqi::mqi_exp<R>(-0.064 * rel.Ek + 7.85 / rel.Ek);
            cs += 9.86;
            cs *= 0.001;
        }
        cs *= mat.rho_mass;
        return cs;
    }

    CUDA_HOST_DEVICE
    virtual void
    along_step(track_t<R>&       trk,
               track_stack_t<R>& stk,
               mqi_rng*          rng,
               const R           len,
               material_t<R>&    mat) {
        ;
    }
};

///< Proton-oxygen inelastic interaction based on tabulated data
template<typename R>
class po_inelastic_tabulated : public po_inelastic<R>
{
public:
    const R* cs_table;
    R        Ek_min;   //= 0.5   ;
    R        Ek_max;   //= 300.0 ;
    R        dEk;      //= 0.5   ;

    R E_bind;
    R E_ratio;
    R E_mini;
    R Prob_2nd;
    R Prob_long;
    R power;

public:
    CUDA_HOST_DEVICE
    po_inelastic_tabulated(R m, R M, R s, const R* p) : cs_table(p) {
        Ek_min    = m;
        Ek_max    = M;
        dEk       = s;
        E_bind    = 5.0;
        E_mini    = 2.0;
        E_ratio   = 0.65;
        Prob_2nd  = 0.65;
        power     = 0.44;
        Prob_long = Prob_2nd + (1 - Prob_2nd) * 0.99;
    }

    CUDA_HOST_DEVICE
    ~po_inelastic_tabulated() {
        cs_table = nullptr;
    }

    CUDA_HOST_DEVICE
    virtual R
    cross_section(const relativistic_quantities<R>& rel, const material_t<R>& mat) {
        R cs = 0;

        if (rel.Ek >= Ek_min && rel.Ek <= Ek_max) {
            uint16_t idx0 = uint16_t((rel.Ek - Ek_min) / dEk);   //0 - 598
            uint16_t idx1 = idx0 + 1;
            R        x0   = Ek_min + idx0 * dEk;
            R        x1   = x0 + 0.5;
            cs            = mqi::intpl1d<R>(rel.Ek, x0, x1, cs_table[idx0], cs_table[idx1]);
        }
        cs *= mat.rho_mass;
        return cs;
    }

    ///< Post-step method to update track's KE, pos, dir, dE, status
    CUDA_HOST_DEVICE
    virtual void
    post_step(track_t<R>&       trk,
              track_stack_t<R>& stk,
              mqi_rng*          rng,
              const R           len,
              material_t<R>&    mat,
              bool              score_local_deposit) {
        const R Ek     = trk.vtx1.ke;
        R       Eb     = this->E_bind;   //Binding energy
        R       Er     = Ek;             // Incident energy to calculate scattering angle
        R       E_2nd  = 0;              // Total energy to secondary proton
        R       E_long = 0;   // Energy loss to long range particle, e.g., leaving enerrgy
        R     E_short = 0;   // Energy loss to short range particle & binding, e.g. locally absorbed
        float dE_total = 0;
        ///< dissipate all energy by looping

        int secondary_protons = 0;

        /// empirical parameters for water I=75 eV
        if (Ek <= 215 && Ek > 200) {
            Prob_2nd = 0.78;
            Prob_long =
              Prob_2nd + (1 - Prob_2nd) * 0.9;   /// P_2nd < P < P_long, 0.93: moqui, 0.4: gpmc
            power = 0.4;
        } else if (Ek > 215) {
            Prob_2nd = 0.78;
            Prob_long =
              Prob_2nd + (1 - Prob_2nd) * 1.0;   /// P_2nd < P < P_long, 0.93: moqui, 0.4: gpmc
            power = 0.4;
        } else if (Ek <= 200 && Ek > 150) {
            Prob_2nd = 0.72;
            Prob_long =
              Prob_2nd + (1 - Prob_2nd) * 0.83;   /// P_2nd < P < P_long, 0.93: moqui, 0.4: gpmc
            power = 0.45;
        } else {
            Prob_2nd = 0.7;
            Prob_long =
              Prob_2nd + (1 - Prob_2nd) * 0.83;   /// P_2nd < P < P_long, 0.93: moqui, 0.4: gpmc
            power = 0.52;
        }
        while ((Er - Eb) > this->E_mini) {
            Er -= Eb;
            ///< sample seconary energy
            R u  = mqi::mqi_uniform<R>(rng);
            R dE = mqi::mqi_pow<R>(u, power) * (Er - this->E_mini) + this->E_mini;
            assert(dE >= 0);
            if (dE >= Er) dE = Er;   //In case deposit energy is greater than E_remain
            Er -= dE;                //substract deposit energy

            ///< energy decrease secondary + binding
            assert(dE + Eb >= 0);
            trk.update_post_vertex_energy(dE + Eb);
            dE_total += (dE + Eb);

            ///< dE can be 2nd-protons, short-range, or long-range
            ///< probability for secondary proton, short-range, and long-range
            R zeta = mqi::mqi_uniform<R>(rng);
            if (zeta < Prob_2nd) {
                /// 50% are secondary proton
                ///< Fippel's paper, probablity of secondary proton is 0.5
                R cos_th = (2.0 * dE / Ek - 1.0) + 2.0 * (1 - dE / Ek) * mqi::mqi_uniform<R>(rng);
                if (cos_th < -1) cos_th = -1;
                if (cos_th > 1) cos_th = 1;
                R th  = mqi::mqi_acos<R>(cos_th);
                R phi = 2.0 * M_PI * mqi::mqi_uniform<R>(rng);

                track_t<R> daughter(trk);
                daughter.dE       = 0;
                daughter.local_dE = 0;
                daughter.primary  = false;
                daughter.process  = mqi::PO_I;
                daughter.vtx0.ke  = dE;
                daughter.vtx1.ke  = dE;
                daughter.status   = CREATED;
                daughter.update_post_vertex_direction(th, phi);
                secondary_protons += 1;

                daughter.vtx0.pos = trk.c_node->geo->rotation_matrix_fwd *
                                      (daughter.vtx1.pos - trk.c_node->geo->translation_vector) +
                                    trk.c_node->geo->translation_vector;
                daughter.vtx0.dir = trk.c_node->geo->rotation_matrix_fwd * daughter.vtx1.dir;
                daughter.vtx1.pos = trk.c_node->geo->rotation_matrix_fwd *
                                      (daughter.vtx1.pos - trk.c_node->geo->translation_vector) +
                                    trk.c_node->geo->translation_vector;
                daughter.vtx1.dir = trk.c_node->geo->rotation_matrix_fwd * daughter.vtx1.dir;
                stk.push_secondary(daughter);
                E_2nd += dE;

            } else if (zeta < Prob_long) {
                ///< we loose energy (dE) as neutron, photon, etc
                ///< 93 % of 50 % long range energy
                E_long += dE;
            } else {
                ///< 7 % of 50 % are short range particles
                ///< locally deposited
#ifdef __PHYSICS_DEBUG__
                track_t<R> daughter(trk);
                daughter.dE       = dE;
                daughter.primary  = false;
                daughter.process  = mqi::PO_I;
                daughter.vtx0.ke  = dE;
                daughter.vtx1.ke  = 0;
                daughter.status   = CREATED;
                daughter.vtx0.pos = trk.c_node->geo->rotation_matrix_fwd *
                                      (daughter.vtx0.pos - trk.c_node->geo->translation_vector) +
                                    trk.c_node->geo->translation_vector;
                daughter.vtx0.dir = trk.c_node->geo->rotation_matrix_fwd * daughter.vtx0.dir;
                daughter.vtx1.pos = trk.c_node->geo->rotation_matrix_fwd *
                                      (daughter.vtx1.pos - trk.c_node->geo->translation_vector) +
                                    trk.c_node->geo->translation_vector;
                daughter.vtx1.dir = trk.c_node->geo->rotation_matrix_fwd * daughter.vtx1.dir;
                stk.push_secondary(daughter);
#else
                if (mat.rho_mass > 1.5e-4f) { trk.local_deposit(dE); }
#endif
                E_short += dE;   //E_short += Eb ;
            }
            /// Deposit binding energy?
            Eb *= E_ratio;   //binding energy changes, 0.7 is tuned value
        }                    //while

        ///< deposit remained energy if any

        ///< locally deposited
        /// Remove in release
        trk.deposit(Er);
        dE_total += Er;
        assert(Er >= 0);
        trk.update_post_vertex_energy(Er);
        trk.stop();
    }
};

}   // namespace mqi

#endif