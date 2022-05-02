#ifndef MQI_NODE_HPP
#define MQI_NODE_HPP

#include <moqui/base/mqi_grid3d.hpp>
#include <moqui/base/mqi_scorer.hpp>

#if defined(__CUDACC__)
#include <cuda_fp16.h>
#endif

/// navigator needs to have "struct" to store those status
/// core-parts should be running in parallel manner, MT, CUDA, OpenCL, FPGA?
/// navigator is shared by block level or device level.
namespace mqi
{

///< node_t : a geometry and it's scorers
/// T: material id
/// R: values in x/y/z and scoreing type
template<typename R>
struct node_t {
    ///< node's geometry
    grid3d<mqi::density_t, R>* geo = nullptr;

    ///< node's scorers
    /// scorer's data, count, mean, and variance need to be allocated seperately
    /// and have corresponding host pointers to download from GPU to CPU.
    uint16_t         n_scorers        = 0;
    scorer<R>**      scorers          = nullptr;
    mqi::key_value** scorers_data     = nullptr;
    mqi::key_value** scorers_count    = nullptr;
    mqi::key_value** scorers_mean     = nullptr;
    mqi::key_value** scorers_variance = nullptr;

    uint16_t           n_children = 0;
    struct node_t<R>** children   = nullptr;
};

}   // namespace mqi
#endif
