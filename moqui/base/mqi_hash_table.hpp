#ifndef MQI_HASH_TABLE_CLASS_HPP
#define MQI_HASH_TABLE_CLASS_HPP

#include <cstring>
#include <moqui/base/mqi_common.hpp>

namespace mqi
{

struct key_value {
    mqi::key_t key1;
    mqi::key_t key2;
    double     value;
};

void
init_table(key_value* table, uint32_t max_capacity) {
    //// Multithreading?
    for (int i = 0; i < max_capacity; i++) {
        table[i].key1  = mqi::empty_pair;
        table[i].key2  = mqi::empty_pair;
        table[i].value = 0;
    }
}

template<typename R>
CUDA_GLOBAL void
init_table_cuda(key_value* table, uint32_t max_capacity) {

    //// Multithreading?
    for (int i = 0; i < max_capacity; i++) {
        table[i].value = 0;
    }
    //#endif
}

template<typename R>
CUDA_GLOBAL void
test_print(mqi::key_value* data) {
    uint32_t ind = 512 * 512 * 200 * 4 - 1;
    printf("data[0].key1 %d data[0].key2 %d data[0].value %d\n",
           data[ind].key1,
           data[ind].key2,
           data[ind].value);
}
}   // namespace mqi
#endif
