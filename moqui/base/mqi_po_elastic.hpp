#ifndef MQI_PO_ELASTIC_HPP
#define MQI_PO_ELASTIC_HPP

#include <moqui/base/mqi_interaction.hpp>

///< Proton-Oxygen elastic interaction
namespace mqi
{

///< Cross-section from Geant4
///< 0.5 MeV 300.0 MeV with 0.5 MeV step
CUDA_CONSTANT const float cs_po_e_g4_table[600] = {
    0.00000, 0.00000, 0.83642, 1.59488, 2.05023, 2.35368, 2.57048, 2.73308, 2.85955, 2.96092,
    3.04356, 3.11282, 3.17103, 3.22122, 3.26438, 3.30252, 3.33597, 3.36575, 3.39252, 3.41593,
    3.43935, 3.45608, 3.47616, 3.49289, 3.50627, 3.52300, 3.53303, 3.54642, 3.58991, 3.63006,
    3.66017, 3.69028, 3.71705, 3.74716, 3.75050, 3.75385, 3.75719, 3.75719, 3.76054, 3.76389,
    3.75050, 3.73712, 3.72708, 3.71370, 3.70366, 3.69363, 3.68024, 3.67021, 3.66017, 3.65013,
    3.63675, 3.62337, 3.60998, 3.59660, 3.58656, 3.57318, 3.56314, 3.55311, 3.53972, 3.52969,
    3.51631, 3.50292, 3.48954, 3.47616, 3.46277, 3.44939, 3.43601, 3.42263, 3.40924, 3.39586,
    3.37913, 3.36240, 3.34367, 3.32627, 3.30887, 3.29148, 3.27408, 3.25668, 3.23928, 3.22189,
    3.19579, 3.16969, 3.14360, 3.11750, 3.09140, 3.06531, 3.03921, 3.01312, 2.98702, 2.96092,
    2.93817, 2.91542, 2.89267, 2.86992, 2.84717, 2.82442, 2.80167, 2.77892, 2.75617, 2.73342,
    2.71134, 2.68925, 2.66717, 2.64509, 2.62301, 2.60093, 2.57885, 2.55677, 2.53468, 2.51260,
    2.49052, 2.46844, 2.44636, 2.42428, 2.40220, 2.38011, 2.35803, 2.33595, 2.31387, 2.29179,
    2.27841, 2.26502, 2.25164, 2.23826, 2.22487, 2.21149, 2.19811, 2.18473, 2.17134, 2.15796,
    2.14458, 2.13120, 2.11781, 2.10443, 2.09105, 2.07766, 2.06428, 2.05090, 2.03752, 2.02413,
    2.00741, 1.99101, 1.97428, 1.95789, 1.94116, 1.92477, 1.90804, 1.89165, 1.87492, 1.85852,
    1.84179, 1.82540, 1.80867, 1.79228, 1.77555, 1.75916, 1.74243, 1.72603, 1.70931, 1.69291,
    1.67786, 1.66280, 1.64775, 1.63269, 1.61763, 1.60258, 1.58752, 1.57247, 1.55741, 1.54236,
    1.52730, 1.51225, 1.49719, 1.48213, 1.46708, 1.45202, 1.43697, 1.42191, 1.40686, 1.39180,
    1.38109, 1.37039, 1.35968, 1.34898, 1.33827, 1.32756, 1.31686, 1.30615, 1.29545, 1.28474,
    1.27403, 1.26333, 1.25262, 1.24191, 1.23121, 1.22050, 1.20980, 1.19909, 1.18838, 1.17768,
    1.16931, 1.16095, 1.15292, 1.14456, 1.13619, 1.12783, 1.11980, 1.11143, 1.10307, 1.09471,
    1.08668, 1.07831, 1.06995, 1.06158, 1.05355, 1.04519, 1.03682, 1.02846, 1.02043, 1.01207,
    1.00370, 0.99534, 0.98731, 0.97894, 0.97058, 0.96222, 0.95419, 0.94582, 0.93746, 0.92909,
    0.92106, 0.91270, 0.90434, 0.89597, 0.88794, 0.87958, 0.87121, 0.86285, 0.85482, 0.84646,
    0.84010, 0.83408, 0.82772, 0.82170, 0.81534, 0.80932, 0.80296, 0.79694, 0.79058, 0.78456,
    0.77820, 0.77218, 0.76583, 0.75980, 0.75345, 0.74742, 0.74107, 0.73504, 0.72869, 0.72267,
    0.71631, 0.71029, 0.70393, 0.69791, 0.69155, 0.68553, 0.67917, 0.67315, 0.66679, 0.66077,
    0.65441, 0.64839, 0.64204, 0.63601, 0.62966, 0.62363, 0.61728, 0.61125, 0.60490, 0.59888,
    0.59553, 0.59218, 0.58884, 0.58549, 0.58215, 0.57880, 0.57546, 0.57211, 0.56876, 0.56542,
    0.56207, 0.55873, 0.55538, 0.55204, 0.54869, 0.54535, 0.54200, 0.53865, 0.53531, 0.53196,
    0.52795, 0.52427, 0.52025, 0.51657, 0.51256, 0.50888, 0.50486, 0.50118, 0.49717, 0.49349,
    0.48947, 0.48579, 0.48178, 0.47810, 0.47408, 0.47040, 0.46639, 0.46271, 0.45869, 0.45501,
    0.45300, 0.45133, 0.44932, 0.44765, 0.44564, 0.44397, 0.44196, 0.44029, 0.43828, 0.43661,
    0.43460, 0.43293, 0.43092, 0.42925, 0.42724, 0.42557, 0.42356, 0.42189, 0.41988, 0.41821,
    0.41620, 0.41453, 0.41252, 0.41085, 0.40884, 0.40717, 0.40516, 0.40349, 0.40148, 0.39981,
    0.39780, 0.39613, 0.39412, 0.39245, 0.39044, 0.38877, 0.38676, 0.38509, 0.38308, 0.38141,
    0.38040, 0.37940, 0.37840, 0.37739, 0.37639, 0.37538, 0.37438, 0.37338, 0.37237, 0.37137,
    0.37037, 0.36936, 0.36836, 0.36736, 0.36635, 0.36535, 0.36434, 0.36334, 0.36234, 0.36133,
    0.36033, 0.35933, 0.35832, 0.35732, 0.35631, 0.35531, 0.35431, 0.35330, 0.35230, 0.35130,
    0.35029, 0.34929, 0.34828, 0.34728, 0.34628, 0.34527, 0.34427, 0.34327, 0.34226, 0.34126,
    0.34092, 0.34059, 0.34026, 0.33992, 0.33959, 0.33925, 0.33892, 0.33858, 0.33825, 0.33791,
    0.33758, 0.33724, 0.33691, 0.33657, 0.33624, 0.33591, 0.33557, 0.33524, 0.33490, 0.33457,
    0.33423, 0.33390, 0.33356, 0.33323, 0.33289, 0.33256, 0.33223, 0.33189, 0.33156, 0.33122,
    0.33089, 0.33055, 0.33022, 0.32988, 0.32955, 0.32921, 0.32888, 0.32855, 0.32821, 0.32788,
    0.32754, 0.32721, 0.32687, 0.32654, 0.32620, 0.32587, 0.32553, 0.32520, 0.32487, 0.32453,
    0.32420, 0.32386, 0.32353, 0.32319, 0.32286, 0.32252, 0.32219, 0.32185, 0.32152, 0.32118,
    0.32085, 0.32052, 0.32018, 0.31985, 0.31951, 0.31918, 0.31884, 0.31851, 0.31817, 0.31784,
    0.31750, 0.31717, 0.31684, 0.31650, 0.31617, 0.31583, 0.31550, 0.31516, 0.31483, 0.31449,
    0.31416, 0.31382, 0.31349, 0.31316, 0.31282, 0.31249, 0.31215, 0.31182, 0.31148, 0.31115,
    0.31081, 0.31048, 0.31014, 0.30981, 0.30947, 0.30914, 0.30881, 0.30847, 0.30814, 0.30780,
    0.30774, 0.30767, 0.30760, 0.30753, 0.30747, 0.30740, 0.30733, 0.30727, 0.30720, 0.30713,
    0.30707, 0.30700, 0.30693, 0.30687, 0.30680, 0.30673, 0.30666, 0.30660, 0.30653, 0.30646,
    0.30640, 0.30633, 0.30626, 0.30620, 0.30613, 0.30606, 0.30600, 0.30593, 0.30586, 0.30579,
    0.30573, 0.30566, 0.30559, 0.30553, 0.30546, 0.30539, 0.30533, 0.30526, 0.30519, 0.30513,
    0.30506, 0.30499, 0.30492, 0.30486, 0.30479, 0.30472, 0.30466, 0.30459, 0.30452, 0.30446,
    0.30439, 0.30432, 0.30426, 0.30419, 0.30412, 0.30406, 0.30399, 0.30392, 0.30385, 0.30379,
    0.30372, 0.30365, 0.30359, 0.30352, 0.30345, 0.30339, 0.30332, 0.30325, 0.30319, 0.30312,
    0.30305, 0.30298, 0.30292, 0.30285, 0.30278, 0.30272, 0.30265, 0.30258, 0.30252, 0.30245,
    0.30238, 0.30232, 0.30225, 0.30218, 0.30211, 0.30205, 0.30198, 0.30191, 0.30185, 0.30178,
    0.30171, 0.30165, 0.30158, 0.30151, 0.30145, 0.30138, 0.30131, 0.30124, 0.30118, 0.30111
};
///< equation
template<typename R>
class po_elastic : public interaction<R, mqi::PROTON>
{
public:
public:
    CUDA_HOST_DEVICE
    virtual R
    cross_section(const relativistic_quantities<R>& rel, const material_t<R>& mat) {
        R cs = 0;

        if (rel.Ek > 50.0 && rel.Ek < 250) {
            cs = 1.88 / rel.Ek;
            cs += 4.0e-5 * rel.Ek;
            cs -= 0.01475;
        }
        //TODO: cs for re.Ek <= 50 ?
        cs *= mat.rho_mass;
        return cs;
    }

    ///< DoIt method to update track's KE, pos, dir, dE, status
    ///< compute energy loss, vertex, secondaries
    CUDA_HOST_DEVICE
    virtual void
    along_step(track_t<R>&       trk,
               track_stack_t<R>& stk,
               mqi_rng*          rng,
               const R           len,
               material_t<R>&    mat) {
        ;
    }

    ///< DoIt method to update track's KE, pos, dir, dE, status
    ///< compute energy loss, vertex, secondaries
    CUDA_HOST_DEVICE
    virtual void
    post_step(track_t<R>&       trk,
              track_stack_t<R>& stk,
              mqi_rng*          rng,
              const R           len,
              material_t<R>&    mat,
              bool              score_local_deposit) {
        mqi::relativistic_quantities<R> rel(trk.vtx1.ke, this->units.Mp);

        if (rel.Ek <= 5.5) {
            R dE = rel.Ek;

            assert(dE >= 0);
#ifdef __PHYSICS_DEBUG__
            /// Remove in release
            track_t<R> daughter(trk);
            daughter.dE       = dE;
            daughter.primary  = false;
            daughter.process  = mqi::PO_E;
            daughter.vtx0.ke  = dE;
            daughter.vtx1.ke  = 0;
            daughter.vtx0.pos = trk.c_node->geo->rotation_matrix_fwd *
                                  (daughter.vtx1.pos - trk.c_node->geo->translation_vector) +
                                trk.c_node->geo->translation_vector;
            daughter.vtx0.dir = trk.c_node->geo->rotation_matrix_fwd * daughter.vtx1.dir;
            daughter.vtx1.pos = trk.c_node->geo->rotation_matrix_fwd *
                                  (daughter.vtx1.pos - trk.c_node->geo->translation_vector) +
                                trk.c_node->geo->translation_vector;
            daughter.vtx1.dir = trk.c_node->geo->rotation_matrix_fwd * daughter.vtx1.dir;
            daughter.status   = CREATED;
            stk.push_secondary(daughter);
#else
            trk.local_deposit(dE);
#endif
            trk.update_post_vertex_energy(dE);
            trk.stop();

        } else {
            R Tp_avg = 0.65 * mqi::mqi_exp(-0.0013 * rel.Ek);
            Tp_avg -= 0.71 * mqi::mqi_exp(-0.0177 * rel.Ek);

            R Tp_max = (2.0 * this->units.Mo * rel.beta_sq * rel.gamma_sq);
            Tp_max /= (1.0 + 2.0 * rel.gamma * this->units.MoMp + this->units.MoMp_sq);

            ///< energy transfered to oxygen and will be locally absorbed.
            R dE = mqi::mqi_exponential<R>(rng, 1.0 / Tp_avg, Tp_max);
            assert(dE >= 0 && dE <= Tp_max);

            R E1      = rel.Ek * (rel.Ek + 2.0 * this->units.Mp);
            R E3      = (rel.Ek - dE) * (rel.Ek - dE + 2.0 * this->units.Mp);
            R cos_th3 = (E1 + E3 - dE * (dE + 2.0 * this->units.Mo)) / 2.0 / mqi::mqi_sqrt(E1 * E3);
            ///< 0<= cos_th3 <= 1
            if (cos_th3 > 1.0) cos_th3 = 1.0;
            if (cos_th3 < -1.0) cos_th3 = -1.0;
            ///< compute scattering matrix
            R th3 = mqi::mqi_acos<R>(cos_th3);
            R phi = 2.0 * M_PI * mqi::mqi_uniform<R>(rng);

            ///< Update Track (P1->P3)
            ///< compute properties at next interaction point

            assert(dE >= 0);
#ifdef __PHYSICS_DEBUG__
            /// Remove in release
            track_t<R> daughter(trk);
            daughter.dE       = dE;
            daughter.primary  = false;
            daughter.process  = mqi::PO_E;
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
            trk.local_deposit(dE);
#endif
            trk.update_post_vertex_energy(dE);
            trk.update_post_vertex_direction(th3, phi);
#if !defined(__CUDACC__)
            if (std::isnan(th3) || std::isnan(phi))
                printf("po_e E1 %f E3 %f rel.Ek %f cos_th3 %f th3 %f phi %f\n",
                       E1,
                       E3,
                       rel.Ek,
                       cos_th3,
                       th3,
                       phi);
            if (std::isnan(trk.vtx1.dir.x) || std::isnan(trk.vtx1.dir.y) ||
                std::isnan(trk.vtx1.dir.z)) {
                printf("inside po_e\n");
                printf("th3 %f phi %f\n", th3, phi);
                printf("po_e ke: %f\n", trk.vtx1.ke);
                printf("1 ");
                trk.vtx0.dir.dump();
                printf("2 ");
                trk.vtx0.pos.dump();
                printf("3 ");
                trk.vtx1.dir.dump();
                printf("4 ");
                trk.vtx1.pos.dump();
            }
            assert(!std::isnan(trk.vtx1.dir.x) && !std::isnan(trk.vtx1.dir.y) &&
                   !std::isnan(trk.vtx1.dir.z));
            assert(!std::isnan(trk.vtx1.pos.x) && !std::isnan(trk.vtx1.pos.y) &&
                   !std::isnan(trk.vtx1.pos.z));
#endif
        }
    }
};

///< proton-oxygen elastic
template<typename R>
class po_elastic_tabulated : public po_elastic<R>
{
public:
    const R* cs_table;
    R        Ek_min = 0.5;
    R        Ek_max = 300.0;
    R        dEk    = 0.5;

public:
    CUDA_HOST_DEVICE
    po_elastic_tabulated(R m, R M, R s, const R* p) : cs_table(p) {
        Ek_min = m;
        Ek_max = M;
        dEk    = s;
    }

    CUDA_HOST_DEVICE
    ~po_elastic_tabulated() {
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
};

}   // namespace mqi
#endif