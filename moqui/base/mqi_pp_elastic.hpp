#ifndef MQI_PP_ELASTIC_HPP
#define MQI_PP_ELASTIC_HPP

#include <moqui/base/mqi_interaction.hpp>

namespace mqi
{

///< Data table (from Geant4 Hard00, Ei=0.5 MeV, Ef = 300.0 MeV, dE = 0.5 MeV,
///< - Cross-section (per volume)
CUDA_CONSTANT const float cs_pp_e_g4_table[600] = {
    0.00000, 2.54109, 4.64796, 4.93231, 4.77709, 4.50612, 4.21575, 3.93742, 3.68050, 3.44700,
    3.23557, 3.04422, 2.87160, 2.71438, 2.57187, 2.44140, 2.32164, 2.21191, 2.11088, 2.01722,
    1.93091, 1.85062, 1.77568, 1.70610, 1.64053, 1.57965, 1.52211, 1.46859, 1.41774, 1.37023,
    1.32474, 1.28192, 1.24178, 1.20364, 1.16684, 1.13272, 1.09993, 1.06849, 1.03905, 1.01028,
    0.98352, 0.95742, 0.93267, 0.90858, 0.88584, 0.86443, 0.84368, 0.82361, 0.80421, 0.78548,
    0.76808, 0.75069, 0.73463, 0.71857, 0.70318, 0.68846, 0.67441, 0.66076, 0.64758, 0.63480,
    0.62249, 0.61058, 0.59901, 0.58790, 0.57713, 0.56669, 0.55659, 0.54676, 0.53732, 0.52809,
    0.51919, 0.51056, 0.50220, 0.49403, 0.48614, 0.47844, 0.47102, 0.46379, 0.45677, 0.44994,
    0.44332, 0.43683, 0.43054, 0.42445, 0.41850, 0.41268, 0.40706, 0.40157, 0.39622, 0.39107,
    0.38598, 0.38103, 0.37621, 0.37153, 0.36698, 0.36250, 0.35815, 0.35393, 0.34978, 0.34577,
    0.34182, 0.33794, 0.33420, 0.33058, 0.32697, 0.32349, 0.32008, 0.31673, 0.31345, 0.31031,
    0.30723, 0.30415, 0.30121, 0.29827, 0.29546, 0.29271, 0.28997, 0.28729, 0.28475, 0.28221,
    0.27973, 0.27726, 0.27492, 0.27257, 0.27030, 0.26809, 0.26595, 0.26381, 0.26167, 0.25966,
    0.25765, 0.25571, 0.25377, 0.25190, 0.25009, 0.24829, 0.24648, 0.24474, 0.24307, 0.24140,
    0.23979, 0.23819, 0.23665, 0.23511, 0.23357, 0.23210, 0.23069, 0.22929, 0.22788, 0.22654,
    0.22521, 0.22387, 0.22260, 0.22139, 0.22012, 0.21892, 0.21771, 0.21657, 0.21544, 0.21430,
    0.21323, 0.21216, 0.21109, 0.21008, 0.20908, 0.20808, 0.20707, 0.20614, 0.20520, 0.20426,
    0.20333, 0.20246, 0.20159, 0.20072, 0.19992, 0.19905, 0.19824, 0.19744, 0.19670, 0.19590,
    0.19516, 0.19443, 0.19369, 0.19296, 0.19229, 0.19162, 0.19095, 0.19028, 0.18961, 0.18901,
    0.18834, 0.18774, 0.18714, 0.18653, 0.18593, 0.18540, 0.18486, 0.18426, 0.18372, 0.18473,
    0.18419, 0.18366, 0.18312, 0.18265, 0.18212, 0.18165, 0.18118, 0.18071, 0.18024, 0.17978,
    0.17931, 0.17891, 0.17844, 0.17804, 0.17764, 0.17723, 0.17683, 0.17643, 0.17603, 0.17563,
    0.17529, 0.17489, 0.17456, 0.17422, 0.17389, 0.17349, 0.17315, 0.17282, 0.17255, 0.17222,
    0.17188, 0.17161, 0.17128, 0.17101, 0.17068, 0.17041, 0.17014, 0.16987, 0.16961, 0.16934,
    0.16907, 0.16880, 0.16854, 0.16827, 0.16807, 0.16780, 0.16760, 0.16733, 0.16713, 0.16686,
    0.16666, 0.16646, 0.16626, 0.16606, 0.16586, 0.16566, 0.16546, 0.16526, 0.16506, 0.16486,
    0.16466, 0.16452, 0.16432, 0.16419, 0.16399, 0.16379, 0.16365, 0.16352, 0.16332, 0.16318,
    0.16305, 0.16285, 0.16272, 0.16258, 0.16245, 0.16231, 0.16218, 0.16205, 0.16191, 0.16178,
    0.16164, 0.16151, 0.16138, 0.16124, 0.16118, 0.16104, 0.16091, 0.16084, 0.16071, 0.16057,
    0.16051, 0.16037, 0.16031, 0.16017, 0.16011, 0.15997, 0.15991, 0.15977, 0.15970, 0.15964,
    0.15950, 0.15944, 0.15937, 0.15930, 0.15917, 0.15910, 0.15904, 0.15897, 0.15890, 0.15883,
    0.15877, 0.15870, 0.15857, 0.15850, 0.15843, 0.15837, 0.15837, 0.15830, 0.15823, 0.15817,
    0.15810, 0.15803, 0.15797, 0.15790, 0.15790, 0.15783, 0.15776, 0.15770, 0.15763, 0.15763,
    0.15756, 0.15750, 0.15750, 0.15743, 0.15736, 0.15736, 0.15730, 0.15723, 0.15723, 0.15716,
    0.15710, 0.15710, 0.15703, 0.15703, 0.15696, 0.15696, 0.15689, 0.15689, 0.15683, 0.15683,
    0.15676, 0.15676, 0.15669, 0.15669, 0.15669, 0.15663, 0.15663, 0.15656, 0.15656, 0.15656,
    0.15649, 0.15649, 0.15643, 0.15643, 0.15643, 0.15636, 0.15636, 0.15636, 0.15629, 0.15629,
    0.15629, 0.15629, 0.15623, 0.15623, 0.15623, 0.15616, 0.15616, 0.15616, 0.15616, 0.15609,
    0.15609, 0.15609, 0.15609, 0.15609, 0.15602, 0.15602, 0.15602, 0.15602, 0.15602, 0.15596,
    0.15596, 0.15596, 0.15596, 0.15596, 0.15596, 0.15589, 0.15589, 0.15589, 0.15589, 0.15589,
    0.15589, 0.15589, 0.15589, 0.15582, 0.15582, 0.15582, 0.15582, 0.15582, 0.15582, 0.15582,
    0.15582, 0.15582, 0.15576, 0.15576, 0.15576, 0.15576, 0.15576, 0.15576, 0.15576, 0.15576,
    0.15576, 0.15576, 0.15576, 0.15576, 0.15576, 0.15576, 0.15576, 0.15569, 0.15569, 0.15569,
    0.15569, 0.15569, 0.15569, 0.15569, 0.15569, 0.15569, 0.15569, 0.15569, 0.15569, 0.15569,
    0.15569, 0.15569, 0.15569, 0.15569, 0.15569, 0.15569, 0.15569, 0.15569, 0.15569, 0.15569,
    0.15569, 0.15569, 0.15569, 0.15569, 0.15569, 0.15569, 0.15569, 0.15569, 0.15569, 0.15569,
    0.15569, 0.15569, 0.15569, 0.15569, 0.15569, 0.15569, 0.15569, 0.15569, 0.15562, 0.15562,
    0.15562, 0.15562, 0.15562, 0.15562, 0.15562, 0.15562, 0.15562, 0.15562, 0.15562, 0.15562,
    0.15562, 0.15562, 0.15562, 0.15562, 0.15562, 0.15562, 0.15562, 0.15562, 0.15562, 0.15562,
    0.15562, 0.15562, 0.15562, 0.15562, 0.15562, 0.15562, 0.15562, 0.15562, 0.15562, 0.15562,
    0.15562, 0.15562, 0.15562, 0.15562, 0.15562, 0.15569, 0.15569, 0.15569, 0.15569, 0.15569,
    0.15569, 0.15569, 0.15569, 0.15569, 0.15569, 0.15569, 0.15569, 0.15569, 0.15569, 0.15569,
    0.15569, 0.15576, 0.15576, 0.15576, 0.15576, 0.15576, 0.15576, 0.15576, 0.15576, 0.15576,
    0.15582, 0.15582, 0.15582, 0.15582, 0.15582, 0.15582, 0.15589, 0.15589, 0.15589, 0.15589,
    0.15589, 0.15589, 0.15596, 0.15596, 0.15596, 0.15596, 0.15596, 0.15602, 0.15602, 0.15602,
    0.15602, 0.15602, 0.15609, 0.15609, 0.15609, 0.15609, 0.15616, 0.15616, 0.15616, 0.15616,
    0.15623, 0.15623, 0.15623, 0.15623, 0.15629, 0.15629, 0.15629, 0.15636, 0.15636, 0.15636,
    0.15636, 0.15643, 0.15643, 0.15643, 0.15649, 0.15649, 0.15649, 0.15656, 0.15656, 0.15656,
    0.15656, 0.15663, 0.15663, 0.15663, 0.15669, 0.15669, 0.15676, 0.15676, 0.15676, 0.15683,
    0.15683, 0.15683, 0.15689, 0.15689, 0.15689, 0.15696, 0.15696, 0.15703, 0.15703, 0.15703
};
///< Proton-Proton elastic interaction
template<typename R>
class pp_elastic : public interaction<R, mqi::PROTON>
{

public:
    ///< DoIt method to update track's KE, pos, dir, dE, status
    ///< compute energy loss, vertex, secondaries
    CUDA_HOST_DEVICE
    virtual void
    along_step(track_t<R>&       trk,
               track_stack_t<R>& stk,
               mqi_rng*          rng,
               const R           len,
               material_t<R>*&   mat,
               R                 rho_mass) {
        ;
    }
};

///< Proton-proton elastic interaction based on tabulated data
template<typename R>
class pp_elastic_tabulated : public pp_elastic<R>
{
public:
    const R* cs_table;
    R        Ek_min = 0.5;
    R        Ek_max = 300.0;
    R        dEk    = 0.5;

public:
    CUDA_HOST_DEVICE
    pp_elastic_tabulated(R m, R M, R s, const R* p) : cs_table(p) {
        Ek_min = m;
        Ek_max = M;
        dEk    = s;
    }

    CUDA_HOST_DEVICE
    ~pp_elastic_tabulated() {
        cs_table = nullptr;
    }

    ///< DoIt method to update track's KE, pos, dir, dE, status
    ///< compute energy loss, vertex, secondaries
    CUDA_HOST_DEVICE
    virtual void
    post_step(track_t<R>&       trk,
              track_stack_t<R>& stk,
              mqi_rng*          rng,
              const R           len,
              material_t<R>*&   mat,
              bool              score_local_deposit) {

        mqi::relativistic_quantities<R> rel(trk.vtx1.ke, this->units.Mp);
        ///< energy transfer ratio
        ///u = 0 no energy transfer
        ///u = 1 100% transfer. this case th1 & th2 may yield nan...
        ///u = 0 case, we need to drop primary
        R min_value = this->Tp_cut / rel.Ek;
        //        R min_value = 0.01 / rel.Ek;
        R u = mqi::mqi_uniform<R>(rng) * (1.0 - 2.0 * min_value) + min_value;
        assert(u < 1);
        ///< From Lorentz invariant for P-P
        ///< P1 : incident particle, P2 : target at rest
        ///< P3 : incident particle, P4 : recoil secondary
        ///< th3 : (P1 - P3)*(P1 - P3) = (P4 - P2)*(P4 - P2)
        ///< th4 : (P1 + P2)*(P1 + P2) = (P3 + P4)*(P3 + P4)
        ///< input energy is vtx1.ke because this is post-step action
        R E1 = rel.Et;
        R dE = rel.Ek * u;
        R E3 = (rel.Ek - dE) + this->units.Mp;
        R E4 = dE + this->units.Mp;
        R P1 = rel.momentum();
        R P3 = mqi::mqi_sqrt(E3 * E3 - this->units.Mp_sq);
        R P4 = mqi::mqi_sqrt(E4 * E4 - this->units.Mp_sq);

        ///< 0<= cos_th3 <= 1
        R cos_th3 = E1 * E3 - this->units.Mp_sq - this->units.Mp * (E1 - E3);
        cos_th3 /= (P1 * P3);
        ///th34 - th3 = th4 (oposite angle against th3)
        R cos_th34 = E3 * E4 - E1 * this->units.Mp;
        cos_th34 /= (P3 * P4);

        if (cos_th3 > 1.0)
            cos_th3 = 1.0;
        else if (cos_th3 < -1.0)
            cos_th3 = -1.0;
        if (cos_th34 > 1.0)
            cos_th34 = 1.0;
        else if (cos_th34 < -1.0)
            cos_th34 = -1.0;

        R th3 = mqi::mqi_acos<R>(cos_th3);
        R th4 = th3 - mqi::mqi_acos<R>(cos_th34);

        ///< phi : angle between scattering plan and reaction plane
        R phi = 2.0 * M_PI * mqi::mqi_uniform<R>(rng);
        assert(dE < trk.vtx1.ke);
        ///< Update Track (P1->P3)
        assert(dE >= 0);
        trk.update_post_vertex_energy(dE);
        trk.update_post_vertex_direction(th3, phi);

        ///< Add a new track (P4)
        ///  P4 is almost same with P3 except KE & DIR
        track_t<R> daughter = trk;
        daughter.dE         = 0;
        daughter.local_dE   = 0;
        daughter.primary    = false;
        daughter.process    = mqi::PP_E;
        daughter.vtx0.ke    = dE;
        daughter.vtx1.ke    = dE;
        daughter.status     = CREATED;
        daughter.update_post_vertex_direction(th4, phi);
        daughter.vtx0.pos = trk.c_node->geo->rotation_matrix_fwd *
                              (daughter.vtx1.pos - trk.c_node->geo->translation_vector) +
                            trk.c_node->geo->translation_vector;
        daughter.vtx0.dir = trk.c_node->geo->rotation_matrix_fwd * daughter.vtx1.dir;
        daughter.vtx1.pos = trk.c_node->geo->rotation_matrix_fwd *
                              (daughter.vtx1.pos - trk.c_node->geo->translation_vector) +
                            trk.c_node->geo->translation_vector;
        daughter.vtx1.dir = trk.c_node->geo->rotation_matrix_fwd * daughter.vtx1.dir;
#if !defined(__CUDACC__)
        if (std::isnan(daughter.vtx1.dir.x) || std::isnan(daughter.vtx1.dir.y) ||
            std::isnan(daughter.vtx1.dir.z)) {
            printf("inside pp_e\n");
            printf("th4 %f phi %f\n", th4, phi);
            printf("po_e ke: %f\n", trk.vtx1.ke);
            printf("1 ");
            daughter.vtx0.dir.dump();
            printf("2 ");
            daughter.vtx0.pos.dump();
            printf("3 ");
            daughter.vtx1.dir.dump();
            printf("4 ");
            daughter.vtx1.pos.dump();
        }
        assert(!std::isnan(daughter.vtx1.dir.x) && !std::isnan(daughter.vtx1.dir.y) &&
               !std::isnan(daughter.vtx1.dir.z));
        assert(!std::isnan(daughter.vtx1.pos.x) && !std::isnan(daughter.vtx1.pos.y) &&
               !std::isnan(daughter.vtx1.pos.z));
        assert(!std::isnan(daughter.vtx0.dir.norm()));
        assert(!std::isnan(daughter.vtx1.dir.norm()));
#endif
        stk.push_secondary(daughter);
    }

    CUDA_HOST_DEVICE
    R
    cross_section(const relativistic_quantities<R>& rel, material_t<R>*& mat, R rho_mass) {
        R cs = 0;

        if (rel.Ek >= Ek_min && rel.Ek <= Ek_max) {
            uint16_t idx0 = uint16_t((rel.Ek - Ek_min) / dEk);   //0 - 598
            uint16_t idx1 = idx0 + 1;
            R        x0   = Ek_min + idx0 * dEk;
            R        x1   = x0 + 0.5;
            cs = mqi::intpl1d<R>(rel.Ek, x0, x1, cs_table[idx0], cs_table[idx1]);
        }
        cs *= rho_mass;
        return cs;
    }
};

}   // namespace mqi

#endif