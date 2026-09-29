#pragma once

#include "carotid_doppler/config.hpp"

#include <cstddef>
#include <stdexcept>

namespace carotid_doppler {

inline std::size_t source_rf_index(
    std::size_t frame,
    std::size_t sequence,
    std::size_t sample,
    std::size_t source_channel,
    std::size_t sequences,
    std::size_t samples,
    std::size_t channels) {
    if (sequence >= sequences || sample >= samples ||
        source_channel >= channels) {
        throw std::out_of_range("source RF coordinate is outside declared shape");
    }
    return (((frame * sequences + sequence) * samples + sample) * channels) +
           source_channel;
}

inline std::size_t internal_rf_index(
    std::size_t frame,
    std::size_t sequence,
    std::size_t internal_channel,
    std::size_t sample,
    std::size_t sequences,
    std::size_t channels,
    std::size_t samples) {
    if (sequence >= sequences || internal_channel >= channels ||
        sample >= samples) {
        throw std::out_of_range(
            "internal RF coordinate is outside declared shape");
    }
    return (((frame * sequences + sequence) * channels + internal_channel) *
            samples) +
           sample;
}

inline std::size_t source_to_internal_channel(
    std::size_t source_channel,
    std::size_t channels,
    ChannelOrder source_order) {
    if (channels == 0 || source_channel >= channels) {
        throw std::out_of_range(
            "source channel is outside declared channel count");
    }
    return source_order == ChannelOrder::descending_x
        ? channels - 1 - source_channel
        : source_channel;
}

}  // namespace carotid_doppler
