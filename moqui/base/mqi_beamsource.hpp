#ifndef MQI_BEAMSOURCE_H
#define MQI_BEAMSOURCE_H

/// \file
///
/// A beamsource is a collection of beamlets and provides an interface for sampling.
#include <map>
#include <moqui/base/mqi_beamlet.hpp>

namespace mqi
{

/// \class beamsource
///
/// \tparam T for types for return values by the distributions
/// \note
template<typename T>
class beamsource
{
public:
    /// A beamlet container.
    /// Each element is a tuple of beamlet, number of histories, and accumlated histories
    std::vector<std::tuple<mqi::beamlet<T>, size_t, size_t>> beamlets_;

    /// A lookup table to map a history to beamlet id.
    /// note: To run GPU, this std::map needs to be replaced GPU compatible data structure.
    std::map<size_t, size_t> cdf2beamlet_;
    size_t*                  array_cdf_;
    size_t*                  array_beamlet_;
    size_t                   beamlet_size_;
    mqi::beamlet<T>*         array_beamlets;
    /// A lookup table to map a time to beamlet id.
    //time,  beamlet_id
    //beamlet_id is -1 for no beam pulse
    std::map<T, int32_t> timeline_;

    /// Default constructor
    beamsource() {
        beamlets_.clear();
        timeline_.clear();
        cdf2beamlet_.clear();
    }

    /// Add a beamlet to internal containers
    /// \param b a beamlet
    /// \param h number of histories
    /// \param p coordinate transformation
    void
    append_beamlet(mqi::beamlet<T>              b,
                   size_t                       h,
                   mqi::coordinate_transform<T> p,
                   T                            time_on  = 1,
                   T                            time_off = 0) {
        b.set_coordinate_transform(p);
        this->append_beamlet(b, h, time_on, time_off);
    }

    /// Add a beamlet to internal containers.
    /// \param b a beamlet
    /// \param h number of histories
    void
    append_beamlet(mqi::beamlet<T> b, size_t h, T time_on = 1, T time_off = 0) {
        const size_t acc = total_histories() + h;
        const size_t beamlet_id =
          this->total_beamlets();   //current number of beamlets -> beamlet ID
        cdf2beamlet_.insert(std::make_pair(acc, beamlet_id));
        beamlets_.push_back(std::make_tuple(b, h, acc));

        T acc_time = this->total_delivery_time() + time_on;
        timeline_.insert(std::make_pair(acc_time, beamlet_id));
        if (time_off != 0) {
            acc_time += time_off;
            timeline_.insert(std::make_pair(acc_time, -1));
        }
    }

    /// Total delivery time
    /// \return end of timeline
    T
    total_delivery_time() const {
        if (timeline_.size() == 0) return 0.0;
        return std::prev(timeline_.end())->first;
    }

    /// Returns size of beamlets.
    /// \return total number of beamlets
    const std::size_t
    total_beamlets() const {
        return beamlets_.size();
    }

    /// Returns total number of histories
    /// \return total number of histories
    const std::size_t
    total_histories() {
        return this->total_beamlets() == 0 ? 0 : std::get<2>(beamlets_.back());
    }

    /// Returns a tuple for given beamlet id.
    /// \return a tuple of beamlet, histories, and accumulated histories
    const std::tuple<mqi::beamlet<T>, size_t, size_t>&
    operator[](unsigned int i) const {
        return beamlets_[i];
    }

    /// Returns a beamlet of a history
    /// \return a beamlet reference (const)
    const mqi::beamlet<T>&
    operator()(size_t h) {
        size_t beamlet_id = cdf2beamlet_.upper_bound(h)->second;
        return std::get<0>(beamlets_[beamlet_id]);
    }

    /// Calculate number of accumulated histories up to given time
    /// \return history as size_t
    /// \param  time
    size_t
    cumulative_history_at_time(T t0) {
        auto pulse = timeline_.lower_bound(t0);
        if (pulse == timeline_.end()) return this->total_histories();

        if (pulse->second == -1) {
            while (pulse->second == -1)
                --pulse;
            return std::get<2>(beamlets_[pulse->second]);
        }

        size_t history = std::get<2>(beamlets_[pulse->second]);

        T pulse_time[2];
        pulse_time[0] = (pulse == timeline_.begin()) ? 0.0 : std::prev(pulse)->first;
        pulse_time[1] = pulse->first;

        const T ratio = (t0 - pulse_time[0]) / (pulse_time[1] - pulse_time[0]);

        if (ratio >= 0) {
            const size_t h0 = std::get<1>(beamlets_[pulse->second]);
            const size_t h1 = (size_t) std::floor((1.0 - ratio) * h0);
            history -= h1;
        }
        return history;
    }
};

}   // namespace mqi

#endif
