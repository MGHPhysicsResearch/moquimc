#ifndef MQI_FIPPEL_PHYSICS_HPP
#define MQI_FIPPEL_PHYSICS_HPP

#include <moqui/base/mqi_error_check.hpp>
#include <moqui/base/mqi_p_ionization.hpp>
#include <moqui/base/mqi_physics_list.hpp>
#include <moqui/base/mqi_po_elastic.hpp>
#include <moqui/base/mqi_po_inelastic.hpp>
#include <moqui/base/mqi_pp_elastic.hpp>

namespace mqi
{

template<typename R>
class fippel_physics : public physics_list<R>
{
public:
    const physics_constants<R>& units = physics_list<R>::units;

    const float max_step        = 1.0;
    const float max_energy_loss = 0.25;   // ratio

    mqi::p_ionization_tabulated<R> p_ion;
    mqi::pp_elastic_tabulated<R>   pp_e;
    mqi::po_elastic_tabulated<R>   po_e;
    mqi::po_inelastic_tabulated<R> po_i;

    CUDA_HOST_DEVICE
    fippel_physics() :
        p_ion(0.1,
              299.6,
              0.5,
              mqi::cs_p_ion_table,
              mqi::restricted_stopping_power_table,
              mqi::range_steps),
        pp_e(0.5, 300.0, 0.5, mqi::cs_pp_e_g4_table), po_e(0.5, 300.0, 0.5, mqi::cs_po_e_g4_table),
        po_i(0.5, 300.0, 0.5, mqi::cs_po_i_g4_table)

    {
        ;
    }

    ///<
    CUDA_HOST_DEVICE
    fippel_physics(R e_cut) :
        physics_list<R>::Te_cut(e_cut), p_ion(0.1,
                                              299.6,
                                              0.5,
                                              mqi::cs_p_ion_table,
                                              mqi::restricted_stopping_power_table,
                                              mqi::range_steps),

        pp_e(0.5, 300.0, 0.5, mqi::cs_pp_e_g4_table), po_e(0.5, 300.0, 0.5, mqi::cs_po_e_g4_table),
        po_i(0.5, 300.0, 0.5, mqi::cs_po_i_g4_table)

    {
        ;
    }

    ///<
    CUDA_HOST_DEVICE
    ~fippel_physics() {
        ;
    }

    ///< step length
    CUDA_HOST_DEVICE
    virtual void
    stepping(track_t<R>&       trk,
             track_stack_t<R>& stk,
             mqi_rng*          rng,
             const R&          rho_mass,
             material_t<R>&    mat,
             const R&          distance_to_boundary,
             bool              score_local_deposit) {

        if (trk.vtx0.ke < this->Tp_cut) {
            if (trk.vtx0.ke < 0) trk.vtx0.ke = 0;
            assert(trk.vtx0.ke >= 0);
            trk.deposit(trk.vtx0.ke);
            trk.update_post_vertex_energy(trk.vtx0.ke);
            p_ion.last_step(trk, mat);
            trk.stop();
            return;
        }

        mqi::relativistic_quantities<R> rel(trk.vtx0.ke, units.Mp);
        R                               length = 0.0;
        ///calculate maximum possible energy-loss
        R max_loss_step    = max_energy_loss * -1.0 * rel.Ek / p_ion.dEdx(rel, mat);
        R current_min_step = this->max_step;
        current_min_step   = current_min_step * mat.stopping_power_ratio(rel.Ek) * mat.rho_mass /
                           this->units.water_density;
        current_min_step  = (current_min_step <= max_loss_step) ? current_min_step : max_loss_step;
        R max_loss_energy = -1.0 * current_min_step * p_ion.dEdx(rel, mat);
        R cs1[4]          = { p_ion.cross_section(rel, mat),
                     pp_e.cross_section(rel, mat),
                     po_e.cross_section(rel, mat),
                     po_i.cross_section(rel, mat) };
        R cs1_sum         = cs1[0] + cs1[1] + cs1[2] + cs1[3];

        mqi::relativistic_quantities<R> rel_de(trk.vtx0.ke - max_loss_energy, units.Mp);
        R                               cs2[4]  = { p_ion.cross_section(rel_de, mat),
                     pp_e.cross_section(rel_de, mat),
                     po_e.cross_section(rel_de, mat),
                     po_i.cross_section(rel_de, mat) };
        R                               cs2_sum = cs2[0] + cs2[1] + cs2[2] + cs2[3];

        ///< Pick bigger cross-section
        R  cs_sum = (cs1_sum >= cs2_sum) ? cs1_sum : cs2_sum;
        R* cs     = (cs1_sum >= cs2_sum) ? cs1 : cs2;

        R prob       = mqi_uniform<R>(rng);           //0-1
        R mfp        = -1.0f * logf(prob) / cs_sum;   // mm, rho_mass is g/mm^3
        R step_limit = current_min_step * this->units.water_density /
                       (mat.stopping_power_ratio(rel.Ek) * mat.rho_mass);
#ifdef DEBUG

        printf("\tcs1: %.3f vs cs2: %.3f, prob: %.3f, mfp: %.3f\n", cs1_sum, cs2_sum, prob, mfp);
        printf(
          "\t%.3f MeV: current_min_step: %.3f, max_loss_step: %.3f, distance_to_boundary: %.3f\n",
          rel.Ek,
          current_min_step,
          max_loss_step,
          distance_to_boundary);
#endif
        ///< Branching depending on distances
        if (distance_to_boundary < mfp && distance_to_boundary < step_limit) {
#ifdef DEBUG
            printf("\td2b: %.3f, ke: %.3f\n", distance_to_boundary, trk.vtx0.ke);
#endif

            assert(!mqi::mqi_isnan(trk.vtx0.ke) && !mqi::mqi_isnan(trk.vtx1.ke));

            p_ion.along_step(trk, stk, rng, distance_to_boundary, mat);

            assert_track<R>(trk, 0);
        } else if ((mfp < distance_to_boundary ||
                    mqi::mqi_abs(mfp - distance_to_boundary) < mqi::geometry_tolerance) &&
                   (mfp < step_limit || mqi::mqi_abs(mfp - step_limit) < mqi::geometry_tolerance)) {
            int p = 0;

#ifdef DEBUG
            printf("\tmfp: %.3f, ke: %.3f, primary:%d\n", mfp, trk.vtx0.ke, trk.primary);
            if (mfp < 0) { printf("mfp: %.3f, ke: %.3f\n", mfp, trk.vtx0.ke); }
#endif

            assert(!mqi::mqi_isnan(trk.vtx0.ke) && !mqi::mqi_isnan(trk.vtx1.ke));
            p_ion.along_step(trk, stk, rng, mfp, mat);
            assert_track<R>(trk, 10);
            if (trk.vtx1.ke < this->Tp_cut) { return; }
            R u          = cs_sum * mqi_uniform<R>(rng);   //0-1
            trk.vtx1.dir = trk.vtx0.dir;
            if (u < cs[0]) {
                p = 0;
                p_ion.post_step(trk, stk, rng, mfp, mat, score_local_deposit);
                assert_track<R>(trk, 1);
            } else if (u < (cs[0] + cs[1])) {
                p = 1;
                pp_e.post_step(trk, stk, rng, mfp, mat, score_local_deposit);
                assert_track<R>(trk, 2);
            } else if (u < (cs[0] + cs[1] + cs[2])) {
                p = 2;
                po_e.post_step(trk, stk, rng, mfp, mat, score_local_deposit);
                assert_track<R>(trk, 3);
            } else if (u < (cs[0] + cs[1] + cs[2] + cs[3])) {
                p = 3;
                po_i.post_step(trk, stk, rng, mfp, mat, score_local_deposit);
                assert_track<R>(trk, 4);
            } else {   //u
            }

#ifdef DEBUG
            if (p != 0) printf("Process: %d\n", p);
#endif
        } else {

#ifdef DEBUG
            if (max_step < 0) {
                printf("max_step: %.3f, ke: %.3f\n", distance_to_boundary, trk.vtx0.ke);
            }
#endif
            assert(!mqi::mqi_isnan(trk.vtx0.ke) && !mqi::mqi_isnan(trk.vtx1.ke));
            p_ion.along_step(trk, stk, rng, step_limit, mat);
            assert_track<R>(trk, 5);
        }

#ifdef DEBUG
        printf("\tend-of-physics:stepping\n");
#endif
        //don't trk.move() here
        return;
    }
};

}   // namespace mqi
#endif