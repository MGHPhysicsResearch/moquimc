#ifndef MQI_ROI_HPP
#define MQI_ROI_HPP
#include <moqui/base/mqi_common.hpp>

namespace mqi
{
typedef enum {
    DIRECT   = 0,   /// scoring index is same in transport index
    INDIRECT = 1,   /// scoring index is stored start[transport_index]
    CONTOUR  = 2    /// scoring index is searched from contour map
} roi_mapping_t;

///< ROI class or struct
/// usage:
/// roi my ;
/// my.idx(v); -> c
//  if roi: length ==1 start = 0, stride = nb_transport_grid, acc_stride == stride
class roi_t
{
public:
    ///< number of pixels of transport grid
    roi_mapping_t method_;
    uint32_t      original_length_;

    ///< determines size of start-lines
    uint32_t length_;

    ///< Following array pointers are set externally once all data uploaded via cudaMemcpy
    uint32_t* start_;        //< where roi-pixels starts
    uint32_t* stride_;       //< number of consecutive pixels
    uint32_t* acc_stride_;   //accumulated stride -> mapped to scorer idx

public:
    CUDA_HOST_DEVICE
    roi_t(roi_mapping_t m,
          uint32_t      n,
          int32_t       l = 0,
          uint32_t*     s = nullptr,
          uint32_t*     t = nullptr,
          uint32_t*     a = nullptr) :
        method_(m),
        original_length_(n), length_(l), start_(s), stride_(t), acc_stride_(a) {
        ;
    }

    CUDA_HOST_DEVICE
    int32_t
    idx(const uint32_t& v) const {

        switch (method_) {
        case INDIRECT:
            return start_[v];
        case CONTOUR:
            return idx_contour(v);
        default:
            return v;
        }
    }

    CUDA_HOST_DEVICE
    int32_t
    get_mask_idx(const uint32_t& v) const {
        switch (method_) {
        case INDIRECT:
            return start_[v];
        case CONTOUR:
            return get_contour_idx(v);
        default:
            return v;
        }
    }

    CUDA_HOST_DEVICE
    int32_t
    get_mask_size() const {
        switch (method_) {
        case INDIRECT:
            return length_;
        case CONTOUR:
            return acc_stride_[length_ - 1];
        default:
            return original_length_;
        }
    }

    CUDA_HOST_DEVICE
    int32_t
    get_contour_idx(const uint32_t& v) const {
        int32_t c = this->lower_bound_cpp(v) - 1;
        if (c < 0) {
            return -1;
        }
        uint32_t distance = v - start_[c];
        if (distance < stride_[c]) {
            /// is in stride
            if (c > 0) distance += acc_stride_[c - 1];
            return distance;
        }
        return -1;   //invalid
    }

    CUDA_HOST_DEVICE
    int32_t
    get_org_idx(const uint32_t& v) const {
        switch (method_) {
        case INDIRECT:
            return start_[v];
        case CONTOUR:
            return get_contour_org(v);
        default:
            return v;
        }
    }

    CUDA_HOST_DEVICE
    int32_t
    get_contour_org(const uint32_t& d) const {
        int32_t  c = this->lower_bound_org(d);
        uint32_t distance;
        if (c > 0) {
            distance = d - acc_stride_[c - 1];
        } else {
            distance = d;
        }
        uint32_t v = start_[c] + distance;
        return v;
    }

    CUDA_HOST_DEVICE
    int32_t
    idx_contour(const uint32_t& v) const {
        int32_t  c        = this->lower_bound_cpp(v) - 1;
        uint32_t distance = v - start_[c];
        if (distance < stride_[c]) {
            /// is in stride
            return 1;
        }
        return -1;   //invalid
    }

    CUDA_HOST_DEVICE
    int32_t
    idx_direct(const uint32_t& v) const {
        return start_[v];
    }

    ///< Binary search
    CUDA_HOST_DEVICE
    int32_t
    lower_bound_cpp(const int32_t& value) const {
        int32_t first = 0;
        int32_t count = length_;
        int32_t step;

        int32_t it;
        while (count > 0) {

            it   = first;
            step = count / 2;
            it += step;

            if (start_[it] <= value) {
                first = ++it;
                count -= step + 1;
            } else
                count = step;
        }
        return first;
    }

    ///< Binary search
    CUDA_HOST_DEVICE
    int32_t
    lower_bound_org(const int32_t& value) const {
        int32_t first = 0;
        int32_t count = length_;
        int32_t step;

        int32_t it;
        while (count > 0) {

            it   = first;
            step = count / 2;
            it += step;

            if (acc_stride_[it] <= value) {
                first = ++it;
                count -= step + 1;
            } else
                count = step;
        }
        return first;
    }
};

}   // namespace mqi

#endif
