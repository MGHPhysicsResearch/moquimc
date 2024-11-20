#ifndef MQI_VARIABLES_HPP
#define MQI_VARIABLES_HPP
#include <moqui/base/mqi_utils.hpp>
#include <moqui/kernel_functions/mqi_kernel_functions.hpp>

///< This file will be deleted soon. (JW: Jan 20, 2021)

namespace mc
{

typedef float            phsp_t;
mqi::material_t<phsp_t>* mc_materials = nullptr;
mqi::node_t<phsp_t>*     mc_world     = nullptr;
mqi::vertex_t<phsp_t>*   mc_vertices  = nullptr;

bool mc_score_variance = true;
}   // namespace mc

#if defined(__CUDACC__)
template<typename R>
CUDA_GLOBAL void
calculate_standard_deviation(mqi::node_t<R>* world,
                             float*          standard_deviation,
                             float*          dose_mean,
                             int             n_histories,
                             int             stat_roi_size,
                             int             c_ind) {
    uint32_t                  thread_id     = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t                  total_threads = (blockDim.x * gridDim.x);
    const mqi::vec2<uint32_t> h_range =
      mqi::start_and_length(total_threads, stat_roi_size, thread_id);
    //    int c_ind = 1;

    uint8_t nb_of_scorers = world->children[c_ind]->n_scorers;
    for (uint32_t i = h_range.x; i < h_range.x + h_range.y; ++i) {
        if (i < stat_roi_size &&
            world->children[c_ind]->scorers[nb_of_scorers - 1]->data_[i].key1 != mqi::empty_pair) {
            standard_deviation[i] = sqrtf(
              ((world->children[c_ind]->scorers[nb_of_scorers - 1]->data_[i].value / n_histories) -
               ((world->children[c_ind]->scorers[nb_of_scorers - 2]->data_[i].value / n_histories) *
                (world->children[c_ind]->scorers[nb_of_scorers - 2]->data_[i].value /
                 n_histories))) /
              (n_histories - 1));
            dose_mean[i] =
              world->children[c_ind]->scorers[nb_of_scorers - 2]->data_[i].value / n_histories;
        }
    }
}

template<typename R>
CUDA_GLOBAL void
calculate_average(mqi::node_t<R>* world, float weight, int roi_size, int c_ind) {
    uint32_t                  thread_id     = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t                  total_threads = (blockDim.x * gridDim.x);
    const mqi::vec2<uint32_t> h_range = mqi::start_and_length(total_threads, roi_size, thread_id);
    //    int                       c_ind   = 1;

    uint8_t nb_of_scorers = world->children[c_ind]->n_scorers;
    for (uint32_t i = h_range.x; i < h_range.x + h_range.y; ++i) {
        for (uint32_t scorer_ind = 0; scorer_ind < world->children[c_ind]->n_scorers - 2;
             scorer_ind++) {
            world->children[c_ind]->scorers[scorer_ind]->data_[i].value =
              world->children[c_ind]->scorers[scorer_ind]->data_[i].value * weight;
        }
    }
}
#endif
#endif   //MQI_VARIABLES_CPP
