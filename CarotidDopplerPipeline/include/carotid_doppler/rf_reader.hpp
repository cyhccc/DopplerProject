#pragma once

#include "carotid_doppler/config.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace carotid_doppler {

class RfInputError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct RfFileFacts {
    std::uintmax_t size_bytes{};
    std::uintmax_t frame_bytes{};
    std::uintmax_t selected_bytes{};
    std::size_t total_frames{};
};

struct MappedRfChunk {
    std::size_t source_frame_start{};
    std::size_t frame_count{};
    std::size_t sequences{};
    std::size_t channels{};
    std::size_t samples{};
    std::vector<std::int16_t> values;
};

struct RfWindowValidation {
    RfFileFacts file;
    std::uint64_t mapped_checksum_fnv1a64{};
    std::uintmax_t mapped_sample_count{};
    bool channel_mapping_bijective{};
    bool sample_multiset_preserved{};
};

RfFileFacts inspect_rf_file(const RfInputConfig& input);

std::vector<std::int16_t> map_source_rf_to_internal(
    const std::vector<std::int16_t>& source_values,
    std::size_t frames,
    std::size_t sequences,
    std::size_t samples,
    std::size_t channels,
    ChannelOrder source_order);

MappedRfChunk read_mapped_rf_chunk(
    const RfInputConfig& input,
    const RfFileFacts& facts,
    std::size_t selected_frame_offset,
    std::size_t frame_count);

RfWindowValidation validate_and_scan_rf_window(
    const RfInputConfig& input,
    std::size_t chunk_frames = 8);

}  // namespace carotid_doppler
