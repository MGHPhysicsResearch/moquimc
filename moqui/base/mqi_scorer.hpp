#ifndef MQI_SCORER_HPP
#define MQI_SCORER_HPP

#include <mutex>

#include <moqui/base/mqi_common.hpp>
#include <moqui/base/mqi_hash_table.hpp>
#include <moqui/base/mqi_material.hpp>
#include <moqui/base/mqi_roi.hpp>

namespace mqi
{
typedef enum {
    VIRTUAL           = 0,
    ENERGY_DEPOSITION = 1,
    DOSE              = 2,   // Dose
    DOSE_Dij          = 3,   //Dose dij matrix
    LETd              = 4,   //Dose weighted LET
    LETt              = 5,   //Track weighted LET
    TRACK_LENGTH      = 6    //Track length
} scorer_t;
///< Foward declerations

// track_t
template<typename R>
class track_t;

// grid3d
template<typename T, typename R>
class grid3d;

template<typename R>
using fp_compute_hit = double (*)(const track_t<R>&,
                                  const mqi::cnb_t&,
                                  grid3d<mqi::density_t, R>&,
                                  mqi::material_t<R>*&);

///< Scorer
template<typename R>
class scorer
{

public:
    ///< Scorer name
    const char* name_;   ///scorer name

    ///< Function pointer for a callback function
    const fp_compute_hit<R> compute_hit_;

    ///< Memory area for scorer data
    mqi::key_value* data_             = nullptr;
    uint32_t        max_capacity_     = 0;   //// Max capacity is 32-bit integer
    uint32_t        current_capacity_ = 0;   //// Max capacity is 32-bit integer

    scorer_t type_;   //< TODO: will be gone

    ///< Region of interest how to map transport pixel to scoring pixel
    roi_t* roi_;

    ///< Variance calculation
    bool save_output = true;

#if defined(__CUDACC__)

#else
    std::mutex mtx;
#endif

    ///< Construct with size
    CUDA_HOST_DEVICE
    scorer(const char*             name,
           const uint32_t          max_capacity,
           const fp_compute_hit<R> func_pointer,
           bool                    save_output = true) :
        name_(name),
        max_capacity_(max_capacity), current_capacity_(max_capacity), compute_hit_(func_pointer),
        save_output(save_output) {
        this->delete_data_if_used();
    }

    CUDA_HOST_DEVICE
    ~scorer() {
        this->delete_data_if_used();
    }

    CUDA_HOST_DEVICE
    void
    delete_data_if_used(void) {
        if (data_ != nullptr) delete[] data_;
        //        if (count_ != nullptr) delete[] count_;
        //        if (mean_ != nullptr) delete[] mean_;
        //        if (variance_ != nullptr) delete[] variance_;
    }
    CUDA_DEVICE
    unsigned long long int
    hash_fun(unsigned long long int k) {
        k ^= k >> 16;
        k *= 0x85ebca6b;
        k ^= k >> 13;
        k *= 0xc2b2ae35;
        k ^= k >> 16;
        return k % (this->max_capacity_ - 1);
    }

    CUDA_HOST_DEVICE
    uint32_t
    CAS(uint32_t* address, uint32_t compare, uint32_t val) {
        uint32_t old = *address;
        if (old == compare) {
            *address = val;
        } else {
        }
        return old;
    }

    CUDA_DEVICE
    void
    insert_pair(mqi::key_t key1, mqi::key_t key2, R value, unsigned long long int scorer_offset) {
        mqi::key_t slot;
        if (key2 == mqi::empty_pair) {
            slot = key1;
            key2 = 0;
        } else {
            slot = hash_fun(key1 + (key2 * scorer_offset));
        }

        uint32_t prev1, prev2;
        while (true) {
#if defined(__CUDACC__)
            prev1 = atomicCAS(&this->data_[slot].key1, mqi::empty_pair, key1);
            prev2 = atomicCAS(&this->data_[slot].key2, mqi::empty_pair, key2);
#else
            prev1 = CAS(&this->data_[slot].key1, mqi::empty_pair, key1);
            prev2 = CAS(&this->data_[slot].key2, mqi::empty_pair, key2);
#endif
            if ((prev1 == mqi::empty_pair || prev1 == key1) &&
                (prev2 == mqi::empty_pair || prev2 == key2)) {
#if defined(__CUDACC__)
                atomicAdd(&this->data_[slot].value, value);
#else
                this->data_[slot].value += value;
#endif
                return;
            }
            slot = (slot + 1) % (this->max_capacity_ - 1);
        }
    }
    //
    //    ///< process hit for Dij matrix?
    //    CUDA_DEVICE
    //    virtual void
    //    process_hit(const track_t<R>&          trk,
    //                const int32_t&             cnb,
    //                grid3d<mqi::density_t, R>& geo,
    //                const uint32_t&            offset,
    //                unsigned long long int     scorer_offset = 0) {
    //        // Calculate index to store hit
    //        // idx : -1 => a hit occured out of ROI. nothing to do.
    //        int32_t idx = roi_->idx(cnb);
    //        if (idx == -1) return;
    //
    //        ///< calculate quantity
    //        R quantity = (*this->compute_hit_)(trk, cnb, geo);
    //
    //        ///< store quantity and variance if it is set.
    //#if defined(__CUDACC__)
    //        insert_pair(cnb, offset, quantity, scorer_offset);
    //
    //        if (this->score_variance_) {
    //            atomicAdd(&count_[cnb].value, 1.0);
    //            R delta = quantity - mean_[cnb].value;
    //            atomicAdd(&mean_[cnb].value, delta / count_[cnb].value);
    //            atomicAdd(&variance_[cnb].value, delta * (quantity - mean_[cnb].value));
    //        }
    //#else
    //        mtx.lock();
    //        insert_pair(cnb, offset, quantity, scorer_offset);
    //        data_[idx].value += quantity;
    //        if (this->score_variance_) {
    //            count_[cnb].value += 1.0;
    //            R delta = quantity - mean_[cnb].value;
    //            mean_[cnb].value += delta / count_[cnb].value;
    //            variance_[cnb].value += delta * (quantity - mean_[cnb].value);
    //        }
    //
    //        mtx.unlock();
    //#endif
    //    }

    ///< clear data
    ///< note: reset data during simulation between runs should called differently
    CUDA_HOST
    void
    clear_data() {
        std::memset(data_, 0xff, sizeof(mqi::key_value) * this->max_capacity_);
        //        if (this->score_variance_) {
        //            std::memset(count_, 0xff, sizeof(mqi::key_value) * this->max_capacity_);
        //            std::memset(mean_, 0xff, sizeof(mqi::key_value) * this->max_capacity_);
        //            std::memset(variance_, 0xff, sizeof(mqi::key_value) * this->max_capacity_);
        //        }
    }
};

}   // namespace mqi

#endif
