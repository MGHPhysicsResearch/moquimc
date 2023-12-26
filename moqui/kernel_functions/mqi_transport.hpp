#ifndef MQI_TRANSPORT_HPP
#define MQI_TRANSPORT_HPP

#include <moqui/base/mqi_error_check.hpp>
#include <moqui/base/mqi_fippel_physics.hpp>
#include <moqui/base/mqi_material.hpp>
#include <moqui/base/mqi_node.hpp>
#include <moqui/base/mqi_threads.hpp>
#include <moqui/base/mqi_track.hpp>
#include <moqui/base/mqi_utils.hpp>
#include <moqui/base/mqi_vertex.hpp>

#include <cassert>

namespace mc
{

//template<typename R> CUDA_GLOBAL void transport_particles_jw(mqi::thrd_t* threads,mqi::node_t<R>* world,mqi::vertex_t<R>*  vertices,const uint32_t n_vtx,uint32_t* tracked_particles,const uint8_t n_th, const uint8_t i_th); // redeclaration may not have initial value

CUDA_DEVICE
unsigned long long int
hash_fun(unsigned long long int k, uint64_t max_capacity) {
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;
    //    return k % (max_capacity - 1);
    return k % (max_capacity);
}

CUDA_DEVICE
uint32_t
hash_fun(uint32_t k1, uint32_t k2, uint64_t max_capacity) {
    k1 *= 0xcc9e2d5;
    k1 = (k1 << 15) | (k1 >> 17);
    k1 *= 0x1b873593;
    k2 ^= k1;
    k2 = (k2 << 13) | (k2 >> 19);
    k2 *= 5;
    k2 += 0xe6546b64;
    k2 ^= 4;
    k2 ^= k2 >> 16;
    k2 *= 0x85ebca6b;
    k2 ^= k2 >> 13;
    k2 *= 0xc2b2ae35;
    k2 ^= k2 >> 16;
    //    return k % (max_capacity - 1);
    return k2 % (max_capacity);
    //    return k2 & (max_capacity - 1);
}

CUDA_HOST_DEVICE
uint32_t
CAS(uint32_t* address, uint32_t compare, uint32_t val) {
    //        printf("old %d compare %d\n", *address, compare);
    uint32_t old = *address;
    if (old == compare) {
        //            printf("Empty\n");
        *address = val;
        //            printf("swap %d\n", *address);
    } else {
        //        printf("Not empty\n");
    }
    return old;
}

template<typename R>
CUDA_DEVICE void
insert_hashtable(mqi::key_value*        hashtable,
                 mqi::key_t             key1,
                 mqi::key_t             key2,
                 double                 value,
                 unsigned long long int scorer_offset,
                 uint64_t               max_capacity) {
    //        mqi::key_t slot = hash(key1 + (key2 * scorer_offset));
    mqi::key_t slot;

    if (value <= 0) {
        return;
    }
    if (key2 == mqi::empty_pair) {
        slot = key1;
        key2 = 0;
    } else {
        slot = hash_fun(key1, key2, max_capacity);
    }

    uint32_t prev1, prev2;
    while (true) {
#if defined(__CUDACC__)
        prev1 = atomicCAS(&hashtable[slot].key1, mqi::empty_pair, key1);
        prev2 = atomicCAS(&hashtable[slot].key2, mqi::empty_pair, key2);
#else
        prev1 = CAS(&hashtable[slot].key1, mqi::empty_pair, key1);
        prev2 = CAS(&hashtable[slot].key2, mqi::empty_pair, key2);
//            printf("prev1 %d prev2 %d\n", prev1, prev2);
#endif
        //            return;
        if ((prev1 == mqi::empty_pair || prev1 == key1) &&
            (prev2 == mqi::empty_pair || prev2 == key2)) {
#if defined(__CUDACC__)
            atomicAdd(&hashtable[slot].value, value);
#else
            hashtable[slot].value += value;
#endif
            return;
        }
        slot = (slot + 1) % (max_capacity);
    }
}

template<typename R>
CUDA_GLOBAL void
transport_particles_patient(mqi::thrd_t*        threads,
                            mqi::node_t<R>*     world,
                            mqi::vertex_t<R>*   vertices,
                            mqi::material_t<R>* material,
                            const uint32_t      n_vtx,
                            uint32_t*           tracked_particles,
                            int32_t*            transport_seed,
                            uint32_t*           scorer_offset_vector = nullptr,
                            bool                score_local_deposit  = true,
                            uint32_t            total_threads        = 1,   // # of CPU threads
                            uint32_t            thread_id            = 0    // CPU thread-id
) {

#if defined(__CUDACC__)
    ///< Thread id and total number of threads are replaced in CUDA
    thread_id     = blockIdx.x * blockDim.x + threadIdx.x;
    total_threads = (blockDim.x * gridDim.x);
#endif

    const mqi::vec2<uint32_t> h_range = mqi::start_and_length(total_threads, n_vtx, thread_id);
    mqi::mqi_rng*          thread_rng = &threads[thread_id].rnd_generator;
    mqi::fippel_physics<R> fippel;
    uint32_t               spot_ind;
    uint32_t               c_ind;
    mqi::vec3<mqi::ijk_t>  index_checker;
    mqi::cnb_t             cnb;   //< child number
    int32_t                ind;
    uint8_t                nb_of_scorers;   //< scorer number
    mqi::track_t<R>       primary;
    mqi::track_stack_t<R> stack;
    ///< count for physics process rates
    for (uint32_t i = h_range.x; i < h_range.x + h_range.y; ++i) {
#if defined(__CUDACC__)
        curand_init(transport_seed[i], 0, 0, thread_rng);   // 57s
#endif
        if (scorer_offset_vector) {
            spot_ind = scorer_offset_vector[i];
        } else {
            spot_ind = mqi::empty_pair;
        }
        primary.set(vertices[i]);
        stack.push_primary(primary);

        ///< do until stacked track is empty
        while (!stack.is_empty()) {
            mqi::track_t<R> track = stack.pop();   // pop a particle
            for (c_ind = 0; c_ind < world->n_children; c_ind++) {
                mqi::grid3d<mqi::density_t, R>& c_geo = *(world->children[c_ind]->geo);
                track.c_node                          = world->children[c_ind];
                nb_of_scorers                         = track.c_node->n_scorers;
                track.vtx0.pos =
                  c_geo.rotation_matrix_inv * (track.vtx0.pos - c_geo.translation_vector);
                track.vtx0.dir = c_geo.rotation_matrix_inv * (track.vtx0.dir);
                track.vtx0.dir.normalize();
                track.vtx1.pos = track.vtx0.pos;
                track.vtx1.dir = track.vtx0.dir;
                index_checker  = c_geo.index(track.vtx0.pos, track.vtx0.dir);
                if (!c_geo.is_valid(index_checker)) {
                    track.its =
                      c_geo.intersect(track.vtx0.pos, track.vtx0.dir);   // The first intersection
                    if (track.its.dist < 0) {
                        track.vtx0.pos =
                          c_geo.rotation_matrix_fwd * (track.vtx0.pos) + c_geo.translation_vector;
                        track.vtx0.dir =
                          c_geo.rotation_matrix_fwd * (track.vtx0.dir);   // rotate the vertex
                        track.vtx1.pos = track.vtx0.pos;
                        track.vtx1.dir = track.vtx0.dir;
                        continue;
                    }
                    track.update_post_vertex_position(track.its.dist);
                    track.move();
                    track.its.cell = c_geo.index(track.vtx0.pos, track.vtx0.dir);
                } else {
                    track.its.dist = 0.0;
                    track.its.cell = index_checker;
                }

                while (c_geo.is_valid(track.its.cell) && !track.is_stopped()) {
                    cnb       = c_geo.ijk2cnb(track.its.cell);
                    track.its = c_geo.intersect(track.vtx0.pos, track.vtx0.dir, track.its.cell);
                    fippel.stepping(track,
                                    stack,
                                    thread_rng,
                                    c_geo[cnb],
                                    material,
                                    track.its.dist,
                                    score_local_deposit);
                    if (track.its.dist < 0) break;

                    for (uint8_t s = 0; s < nb_of_scorers - 2; ++s) {
                        if (track.c_node->scorers[s]->roi_->idx(cnb) > 0) {
                            insert_hashtable<R>(
                              track.c_node->scorers[s]->data_,
                              cnb,
                              spot_ind,
                              track.c_node->scorers[s]->compute_hit_(track, cnb, c_geo, material),
                              c_geo.get_nxyz().x * c_geo.get_nxyz().y * c_geo.get_nxyz().z,
                              track.c_node->scorers[s]->max_capacity_);
                        }
                    }
                    for (uint8_t s = 0; s < nb_of_scorers; ++s) {
                        if (track.c_node->scorers[s]->roi_->idx(cnb) > 0) {
                            insert_hashtable<R>(
                              track.c_node->scorers[s]->data_,
                              cnb,
                              spot_ind,
                              track.c_node->scorers[s]->compute_hit_(track, cnb, c_geo, material),
                              c_geo.get_nxyz().x * c_geo.get_nxyz().y * c_geo.get_nxyz().z,
                              track.c_node->scorers[s]->max_capacity_);
                        }
                    }

                    if (!track.is_stopped()) {
                        c_geo.index(track.vtx1.pos,
                                    track.vtx1.dir,
                                    track.its.cell);   // update the cell index of the particle
                        track.move();
                    }
                }
                track.vtx0.pos =
                  c_geo.rotation_matrix_fwd * (track.vtx0.pos) + c_geo.translation_vector;
                track.vtx0.dir =
                  c_geo.rotation_matrix_fwd * (track.vtx0.dir);   // rotate the vertex
                track.vtx1.pos = track.vtx0.pos;
                track.vtx1.dir = track.vtx0.dir;
            }   //while(history is out-of-world or zero energy

        }   //while(stack is not empty)

#if defined(__CUDACC__)
        atomicAdd(tracked_particles, 1);
#else
        tracked_particles[0] += 1;
#endif
    }   //for
}   //transport_particles_table

template<typename R>
CUDA_GLOBAL void
transport_particles_patient_stat(mqi::thrd_t*        threads,
                                 mqi::node_t<R>*     world,
                                 mqi::vertex_t<R>*   vertices,
                                 mqi::material_t<R>* material,
                                 const uint32_t      n_vtx,
                                 uint32_t*           tracked_particles,
                                 int32_t*            transport_seed,
                                 uint32_t*           scorer_offset_vector = nullptr,
                                 bool                score_local_deposit  = true,
                                 uint32_t            total_threads        = 1,   // # of CPU threads
                                 uint32_t            thread_id            = 0    // CPU thread-id
) {

#if defined(__CUDACC__)
    ///< Thread id and total number of threads are replaced in CUDA
    thread_id     = blockIdx.x * blockDim.x + threadIdx.x;
    total_threads = (blockDim.x * gridDim.x);
#endif

    const mqi::vec2<uint32_t> h_range = mqi::start_and_length(total_threads, n_vtx, thread_id);

    mqi::mqi_rng*          thread_rng = &threads[thread_id].rnd_generator;
    mqi::fippel_physics<R> fippel;
    uint32_t               spot_ind;
    uint32_t               c_ind;
    mqi::vec3<mqi::ijk_t>  index_checker;
    mqi::cnb_t             cnb;   //< child number
    int32_t                ind;
    uint8_t                nb_of_scorers;   //< scorer number
    mqi::track_t<R>       primary;
    mqi::track_stack_t<R> stack;
    ///< count for physics process rates
    for (uint32_t i = h_range.x; i < h_range.x + h_range.y; ++i) {
#if defined(__CUDACC__)
        curand_init(transport_seed[i], 0, 0, thread_rng);   // 57s
#endif
        if (scorer_offset_vector) {
            spot_ind = scorer_offset_vector[i];
        } else {
            spot_ind = mqi::empty_pair;
        }
        primary.set(vertices[i]);
        stack.push_primary(primary);

        ///< do until stacked track is empty
        while (!stack.is_empty()) {
            mqi::track_t<R> track = stack.pop();   // pop a particle
            for (c_ind = 0; c_ind < world->n_children; c_ind++) {
                mqi::grid3d<mqi::density_t, R>& c_geo = *(world->children[c_ind]->geo);
                track.c_node                          = world->children[c_ind];
                nb_of_scorers                         = track.c_node->n_scorers;
                track.vtx0.pos =
                  c_geo.rotation_matrix_inv * (track.vtx0.pos - c_geo.translation_vector);
                track.vtx0.dir = c_geo.rotation_matrix_inv * (track.vtx0.dir);
                track.vtx0.dir.normalize();
                track.vtx1.pos = track.vtx0.pos;
                track.vtx1.dir = track.vtx0.dir;
                index_checker  = c_geo.index(track.vtx0.pos, track.vtx0.dir);
                if (!c_geo.is_valid(index_checker)) {
                    track.its =
                      c_geo.intersect(track.vtx0.pos, track.vtx0.dir);   // The first intersection
                    if (track.its.dist < 0) {
                        track.vtx0.pos =
                          c_geo.rotation_matrix_fwd * (track.vtx0.pos) + c_geo.translation_vector;
                        track.vtx0.dir =
                          c_geo.rotation_matrix_fwd * (track.vtx0.dir);   // rotate the vertex
                        track.vtx1.pos = track.vtx0.pos;
                        track.vtx1.dir = track.vtx0.dir;
                        continue;
                    }
                    track.update_post_vertex_position(track.its.dist);
                    track.move();
                    track.its.cell = c_geo.index(track.vtx0.pos, track.vtx0.dir);
                } else {
                    track.its.dist = 0.0;
                    track.its.cell = index_checker;
                }

                while (c_geo.is_valid(track.its.cell) && !track.is_stopped()) {
                    cnb       = c_geo.ijk2cnb(track.its.cell);
                    track.its = c_geo.intersect(track.vtx0.pos, track.vtx0.dir, track.its.cell);
                    fippel.stepping(track,
                                    stack,
                                    thread_rng,
                                    c_geo[cnb],
                                    material,
                                    track.its.dist,
                                    score_local_deposit);
                    if (track.its.dist < 0) break;

                    for (uint8_t s = 0; s < nb_of_scorers - 2; ++s) {
                        if (track.c_node->scorers[s]->roi_->idx(cnb) > 0) {
                            insert_hashtable<R>(
                              track.c_node->scorers[s]->data_,
                              cnb,
                              spot_ind,
                              track.c_node->scorers[s]->compute_hit_(track, cnb, c_geo, material),
                              c_geo.get_nxyz().x * c_geo.get_nxyz().y * c_geo.get_nxyz().z,
                              track.c_node->scorers[s]->max_capacity_);
                        }
                    }

                    for (uint8_t s = nb_of_scorers - 2; s < nb_of_scorers; ++s) {
                        ind = track.c_node->scorers[s]->roi_->get_mask_idx(cnb);
                        if (ind >= 0) {
                            insert_hashtable<R>(
                              track.c_node->scorers[s]->data_,
                              (uint32_t) ind,
                              spot_ind,
                              track.c_node->scorers[s]->compute_hit_(track, cnb, c_geo, material),
                              c_geo.get_nxyz().x * c_geo.get_nxyz().y * c_geo.get_nxyz().z,
                              track.c_node->scorers[s]->max_capacity_);
                        }
                    }

                    if (!track.is_stopped()) {
                        c_geo.index(track.vtx1.pos,
                                    track.vtx1.dir,
                                    track.its.cell);   // update the cell index of the particle
                        track.move();
                    }
                }
                track.vtx0.pos =
                  c_geo.rotation_matrix_fwd * (track.vtx0.pos) + c_geo.translation_vector;
                track.vtx0.dir =
                  c_geo.rotation_matrix_fwd * (track.vtx0.dir);   // rotate the vertex
                track.vtx1.pos = track.vtx0.pos;
                track.vtx1.dir = track.vtx0.dir;
            }   //while(history is out-of-world or zero energy

        }   //while(stack is not empty)

#if defined(__CUDACC__)
        atomicAdd(tracked_particles, 1);
#else
        tracked_particles[0] += 1;
#endif
    }   //for
}   //transport_particles_table
}   // namespace mc

#endif
