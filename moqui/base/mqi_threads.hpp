#ifndef MQI_THREAD_HPP
#define MQI_THREAD_HPP

#include <moqui/base/mqi_common.hpp>
#include <moqui/base/mqi_math.hpp>

namespace mqi
{

///< thread struct to hold random generator
///< thread local
struct thrd_t {
    uint32_t histories[2];    // histories from and to
    mqi_rng  rnd_generator;   //
};

///< random number initialization before entering the  mc loop.
///< this can be as a part of the loop also.
CUDA_GLOBAL
void
initialize_threads(mqi::thrd_t*   thrds,
                   const uint32_t n_threads,
                   unsigned long  master_seed = 0,
                   unsigned long  offset      = 0) {
#if defined(__CUDACC__)
    uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(master_seed + blockIdx.x, threadIdx.x, offset, &thrds[thread_id].rnd_generator);
#else
    for (uint32_t i = 0; i < n_threads; ++i) {
        std::seed_seq seed{ master_seed + i };
        thrds[i].rnd_generator.seed(master_seed);
    }
#endif
}

}   // namespace mqi

#endif
