#ifndef MQI_DOWNLOAD_DATA_HPP
#define MQI_DOWNLOAD_DATA_HPP

#include <fstream>
#include <iostream>
#include <moqui/base/mqi_node.hpp>
#include <moqui/base/mqi_scorer.hpp>
#include <moqui/kernel_functions/mqi_variables.hpp>

namespace mc
{

#if defined(__CUDACC__)
template<typename R>
CUDA_GLOBAL void
download_scorers(mqi::node_t<R>* g_node_mother, uint16_t g_node_id, uint16_t scorer_id, R* deposit);
template<typename R>
void
download_node(mqi::node_t<R>* c_node, mqi::node_t<R>*& g_node);
#endif

#if defined(__CUDACC__)
template<typename R>
CUDA_GLOBAL void
download_scorers(mqi::node_t<R>* g_node_mother,
                 uint16_t        g_node_id,
                 uint16_t        scorer_id,
                 R*              deposit) {
    g_node_mother->children[g_node_id]->scorers[scorer_id]->data_ = deposit;
}

///< Download nodes from GPU to CPU
///< recursive operation
///< it just needs to download scorers. we don't have to download images
template<typename R>
void
download_node(mqi::node_t<R>* c_node, mqi::node_t<R>*& g_node) {
    mqi::node_t<R> tmp;   ///< copy of device node
    gpu_err_chk(cudaMemcpy(&tmp, g_node, sizeof(mqi::node_t<R>), cudaMemcpyDeviceToHost));

    //printf("node:%p, n_scorers: %d, n_children: %d, *p children:%p, *p scorers:%p\n",
    //       g_node, tmp.n_scorers, tmp.n_children, tmp.children, tmp.scorers_data);

    printf("n_scorers: %d, %p\n", c_node->n_scorers, tmp.scorers_data);

    if (tmp.n_scorers > 0) {
        printf("c_node %d\n", c_node->scorers[0]->max_capacity_);
        mqi::key_value** scrs        = new mqi::key_value*[tmp.n_scorers];
        mqi::key_value** scors_count = nullptr;
        mqi::key_value** scors_mean  = nullptr;
        mqi::key_value** scors_var   = nullptr;
        //        tmp.scorers                                 = new mqi::v_scorer<R>*[tmp.n_scorers];
        if (mc::mc_score_variance) {
            scors_count = new mqi::key_value*[tmp.n_scorers];
            scors_mean  = new mqi::key_value*[tmp.n_scorers];
            scors_var   = new mqi::key_value*[tmp.n_scorers];
        }

        gpu_err_chk(cudaMemcpy(
          scrs, tmp.scorers_data, tmp.n_scorers * sizeof(mqi::key_value*), cudaMemcpyDeviceToHost));
        gpu_err_chk(cudaFree(tmp.scorers_data));

//        if (mc::mc_score_variance) {
//            gpu_err_chk(cudaMemcpy(scors_count,
//                                   tmp.scorers_count,
//                                   tmp.n_scorers * sizeof(mqi::key_value*),
//                                   cudaMemcpyDeviceToHost));
//            gpu_err_chk(cudaMemcpy(scors_mean,
//                                   tmp.scorers_mean,
//                                   tmp.n_scorers * sizeof(mqi::key_value*),
//                                   cudaMemcpyDeviceToHost));
//            gpu_err_chk(cudaMemcpy(scors_var,
//                                   tmp.scorers_variance,
//                                   tmp.n_scorers * sizeof(mqi::key_value*),
//                                   cudaMemcpyDeviceToHost));
//        }

        for (int i = 0; i < tmp.n_scorers; ++i) {
            printf("scrs[%d] : %p\n", i, scrs[i]);
            gpu_err_chk(cudaMemcpy(c_node->scorers[i]->data_,
                                   scrs[i],
                                   c_node->scorers[i]->max_capacity_ * sizeof(mqi::key_value),
                                   cudaMemcpyDeviceToHost));
            gpu_err_chk(cudaFree(scrs[i]));
//            if (c_node->scorers[i]->score_variance_) {
//                gpu_err_chk(cudaMemcpy(c_node->scorers[i]->count_,
//                                       scors_count[i],
//                                       c_node->scorers[i]->max_capacity_ * sizeof(mqi::key_value),
//                                       cudaMemcpyDeviceToHost));
//                gpu_err_chk(cudaMemcpy(c_node->scorers[i]->mean_,
//                                       scors_mean[i],
//                                       c_node->scorers[i]->max_capacity_ * sizeof(mqi::key_value),
//                                       cudaMemcpyDeviceToHost));
//
//                gpu_err_chk(cudaMemcpy(c_node->scorers[i]->variance_,
//                                       scors_var[i],
//                                       c_node->scorers[i]->max_capacity_ * sizeof(mqi::key_value),
//                                       cudaMemcpyDeviceToHost));
//            }
        }
        delete[] scrs;
//        delete[] scors_count;
//        delete[] scors_mean;
//        delete[] scors_var;
    }
    if (tmp.n_children > 0) {
        mqi::node_t<R>** children = new mqi::node_t<R>*[tmp.n_children];
        gpu_err_chk(cudaMemcpy(children,
                               tmp.children,
                               tmp.n_children * sizeof(mqi::node_t<R>*),
                               cudaMemcpyDeviceToHost));

        for (int i = 0; i < tmp.n_children; ++i) {
            printf("\tnode's child[%d]: %p\n", i, children[i]);
            download_node<R>(c_node->children[i], children[i]);
        }
        //        delete[] children;
        //        gpu_err_chk(cudaFree(tmp.children));
    }
    //    gpu_err_chk(cudaFree(g_node));
}
#endif
}   // namespace mc
#endif   //DOWNLOAD_DATA_CPP
