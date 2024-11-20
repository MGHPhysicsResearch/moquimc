
#ifndef MQI_TRACK_STACK_HPP
#define MQI_TRACK_STACK_HPP

#include <moqui/base/mqi_node.hpp>
#include <moqui/base/mqi_track.hpp>

namespace mqi
{

template<typename R>
class track_stack_t
{

public:
#ifdef __PHYSICS_DEBUG__
    const uint16_t limit = 200;
    track_t<R>     tracks[200];
#else
    const uint16_t limit = 10;
    track_t<R>     tracks[10];
#endif
    uint16_t idx = 0;   /// empty : 0, 1-st element : 1
    CUDA_HOST_DEVICE
    track_stack_t() {
        ;
    }

    CUDA_HOST_DEVICE
    ~track_stack_t() {
        ;
    }

    CUDA_HOST_DEVICE
    void
    push_secondary(const track_t<R>& trk) {
        if (idx < limit) {
            tracks[idx] = trk;
            ++idx;
        }
    }

    CUDA_HOST_DEVICE
    void
    push_primary(const track_t<R>& trk) {
        tracks[0] = trk;
        idx       = 1;
    }
    CUDA_HOST_DEVICE
    bool
    is_empty(void) {
        return idx == 0;
    }

    CUDA_HOST_DEVICE
    track_t<R>
    pop(void) {
        ///copy
        return tracks[--idx];
    }

    CUDA_HOST_DEVICE
    track_t<R>&
    operator[](uint16_t i) {
        return tracks[i];
    }
};

}   // namespace mqi
#endif
