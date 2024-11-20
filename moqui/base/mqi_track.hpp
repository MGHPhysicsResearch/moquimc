#ifndef MQI_TRACK_HPP
#define MQI_TRACK_HPP

#include <moqui/base/mqi_node.hpp>
#include <moqui/base/mqi_vertex.hpp>

namespace mqi
{

///< Particle type
typedef enum {
    PHOTON   = 0,
    ELECTRON = 1,
    PROTON   = 2,
    NEUTRON  = 3
} particle_t;
//PDG[PHOTON] =
//PDG[PROTON] = 2212 //

///< Process type
typedef enum {
    BEGIN    = 0,
    MAX_STEP = 1,
    BOUNDARY = 2,
    CSDA     = 3,
    D_ION    = 4,
    PP_E     = 5,
    PO_E     = 6,
    PO_I     = 7,
    KILLED0  = 8,
    KILLED1  = 9
} process_t;

///< status of particle
///< CREATED : just created
///< ALIVE   : under tracking
///< STOPPED : no further tracking is required due to limitation, e.g., cut
///< KILLED  : no further tracking is required due to energy 0 or exit world boundary?
typedef enum {
    CREATED = 0,   ///< created
    STOPPED = 3    ///< stopped by physics process
} status_t;

///< Track class
/// pointer to geometry where current track is placed
/// pre-/post- vertex points: vtx0 vtx1

template<typename R>
class track_t
{
public:
    uint32_t  scorer_column;   ///< id of beamlet or time-index can be used for Dij or Dit matrix
    status_t  status;          ///< particle status
    process_t process;         ///< id of physics process that limit the step,
    ///< -1: geometry, 0: CSDA, 1: delta, 2: pp-e, 3: po-e, 4: po-i
    bool       primary;
    particle_t particle;   ///< particle type,

    vertex_t<R> vtx0;   ///< vertex pre
    vertex_t<R> vtx1;   ///< vertex post

    R          dE       = 0.0;       ///< total energy deposit between vtx0 to vtx1
    R          local_dE = 0.0;       ///< total energy deposit between vtx0 to vtx1
    node_t<R>* c_node   = nullptr;   ///< current node

    intersect_t<R> its;   ///< geometry information, intersection, copy number

    vec3<R> ref_vector = vec3<R>(0, 0, 1);
    ///< Defaut constructor
    CUDA_HOST_DEVICE
    track_t() :
        status(CREATED), process(BEGIN), primary(true), dE(0), scorer_column(0), local_dE(0) {
        ;
    }

    ///< Constructor
    CUDA_HOST_DEVICE
    track_t(const vertex_t<R>& v) :
        status(CREATED), process(BEGIN), primary(true), dE(0), scorer_column(0), local_dE(0) {
        vtx0 = v;
        vtx1 = v;
    }

    ///< Constructor
    CUDA_HOST_DEVICE
    track_t(status_t    s,
            process_t   p,
            bool        is_p,
            particle_t  t,
            vertex_t<R> v0,
            vertex_t<R> v1,
            const R&    dE) :
        status(s),
        process(p), primary(is_p), particle(t), vtx0(v0), vtx1(v1), dE(dE), scorer_column(0),
        local_dE(0) {
        ;
    }

    ///< copy constructor
    CUDA_HOST_DEVICE
    track_t(const track_t& rhs) {
        scorer_column = rhs.scorer_column;
        status        = rhs.status;
        process       = rhs.process;
        particle      = rhs.particle;
        primary       = rhs.primary;
        vtx0          = rhs.vtx0;
        vtx1          = rhs.vtx1;
        dE            = rhs.dE;
        local_dE      = rhs.local_dE;
        c_node        = rhs.c_node;
        its           = rhs.its;
        ref_vector    = rhs.ref_vector;
    }

    ///< Destructor
    CUDA_HOST_DEVICE
    ~track_t() {
        ;
    }

    CUDA_HOST_DEVICE
    void
    set(const vertex_t<R>& v) {
        this->status        = CREATED;
        this->process       = BEGIN;
        this->primary       = true;
        this->dE            = 0;
        this->scorer_column = 0;
        this->local_dE      = 0;
        vtx0                = v;
        vtx1                = v;
    }
    ///< Deposit energy
    CUDA_HOST_DEVICE
    void
    deposit(R e) {
        dE += e;
    }

    ///< Deposit energy
    CUDA_HOST_DEVICE
    void
    local_deposit(R e) {
        local_dE += e;
    }

    CUDA_HOST_DEVICE
    bool
    is_stopped() {
        return status == STOPPED;
    }

    ///< Update vertex point for given R
    CUDA_HOST_DEVICE
    void
    shorten_step(R ratio)   //0 < ratio < 1
    {
        vtx1.pos = vtx0.pos + (vtx1.pos - vtx0.pos) * ratio;
    }

    ///<
    CUDA_HOST_DEVICE
    void
    update_post_vertex_direction(const R& theta, const R& phi) {
        mqi::mat3x3<R> m_local(0, theta, phi);
        mqi::vec3<R>   d_local = m_local * ref_vector;   // rotate about the z-axis (dir)
        d_local.normalize();
        mqi::mat3x3<R> m_global(ref_vector, vtx1.dir);   // match dir to vtx1.dir
        vtx1.dir = m_global * d_local;
        vtx1.dir.normalize();
    }

    ///< called by CSDA.
    ///< no needs to get called by nuclear interactions
    CUDA_HOST_DEVICE
    void
    update_post_vertex_position(const R& len) {
        vtx1.pos = vtx0.pos + vtx0.dir * len;
    }

    ///<
    CUDA_HOST_DEVICE
    void
    update_post_vertex_energy(const R& e) {
        vtx1.ke -= e;
    }

    ///< Proceed a step
    ///< vtx0 = vtx1
    CUDA_HOST_DEVICE
    void
    move() {
        vtx0     = vtx1;
        dE       = 0;
        local_dE = 0;
    }

    ///< Change particle status
    CUDA_HOST_DEVICE
    void
    stop() {
        status = STOPPED;
    }
};

///< Check invalid direction in track
template<typename R>
CUDA_HOST_DEVICE void
assert_track(const mqi::track_t<R>& trk, int8_t id = -1) {
    if (mqi::mqi_isnan(trk.vtx1.dir.x) || mqi::mqi_isnan(trk.vtx1.dir.y) ||
        mqi::mqi_isnan(trk.vtx1.dir.z)) {
        printf("id: %d\n", id);
        printf("vtx0.pos ");
        trk.vtx0.pos.dump();
        printf("vtx0.dir ");
        trk.vtx0.dir.dump();
        printf("vtx1.pos ");
        trk.vtx1.pos.dump();
        printf("vtx1.dir ");
        trk.vtx1.dir.dump();
        printf("There is nan in track direction\n");
        exit(1);
    } else if (trk.vtx0.dir.norm() < mqi::geometry_tolerance ||
               trk.vtx1.dir.norm() < mqi::geometry_tolerance) {
        printf("id: %d\n", id);
        printf("vtx0.pos ");
        trk.vtx0.pos.dump();
        printf("vtx0.dir ");
        trk.vtx0.dir.dump();
        printf("vtx1.pos ");
        trk.vtx1.pos.dump();
        printf("vtx1.dir ");
        trk.vtx1.dir.dump();
        printf("There is all-zeros in track direction\n");
        exit(1);
    }
}

}   // namespace mqi
#endif
