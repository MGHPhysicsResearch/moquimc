#ifndef MQI_VARIABLES_HPP
#define MQI_VARIABLES_HPP
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
#endif   //MQI_VARIABLES_CPP
