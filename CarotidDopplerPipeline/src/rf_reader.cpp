#include "carotid_doppler/rf_reader.hpp"

#include "carotid_doppler/data_layout.hpp"

#include <algorithm>
#include <bit>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace carotid_doppler {
namespace {

constexpr std::uint64_t kFnvOffsetBasis = 14695981039346656037ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

template <typename Value>
Value checked_multiply(Value left, Value right, const char* description) {
    if (left != 0 && right > std::numeric_limits<Value>::max() / left) {
        throw RfInputError(std::string(description) + " overflows");
    }
    return left * right;
}

template <typename Value>
Value checked_add(Value left, Value right, const char* description) {
    if (right > std::numeric_limits<Value>::max() - left) {
        throw RfInputError(std::string(description) + " overflows");
    }
    return left + right;
}

std::size_t samples_per_frame(const RfInputConfig& input) {
    std::size_t count =
        checked_multiply(input.sequences, input.samples, "RF samples per frame");
    return checked_multiply(count, input.channels, "RF samples per frame");
}

std::uintmax_t bytes_for_samples(
    std::uintmax_t sample_count,
    const char* description) {
    return checked_multiply(
        sample_count,
        static_cast<std::uintmax_t>(sizeof(std::int16_t)),
        description);
}

std::streamoff checked_stream_offset(std::uintmax_t byte_offset) {
    if (byte_offset >
        static_cast<std::uintmax_t>(std::numeric_limits<std::streamoff>::max())) {
        throw RfInputError("RF byte offset exceeds stream offset range");
    }
    return static_cast<std::streamoff>(byte_offset);
}

std::streamsize checked_stream_size(std::uintmax_t byte_count) {
    if (byte_count >
        static_cast<std::uintmax_t>(std::numeric_limits<std::streamsize>::max())) {
        throw RfInputError("RF read size exceeds stream size range");
    }
    return static_cast<std::streamsize>(byte_count);
}

bool channel_mapping_is_bijective(
    std::size_t channels,
    ChannelOrder source_order) {
    std::vector<bool> visited(channels, false);
    for (std::size_t source_channel = 0; source_channel < channels;
         ++source_channel) {
        const std::size_t internal_channel = source_to_internal_channel(
            source_channel, channels, source_order);
        if (visited[internal_channel]) {
            return false;
        }
        visited[internal_channel] = true;
    }
    return std::all_of(
        visited.begin(), visited.end(), [](bool present) { return present; });
}

std::uint64_t update_checksum(
    std::uint64_t checksum,
    const std::vector<std::int16_t>& values) {
    for (const std::int16_t value : values) {
        const auto bits = static_cast<std::uint16_t>(value);
        checksum ^= static_cast<std::uint8_t>(bits & 0xFFU);
        checksum *= kFnvPrime;
        checksum ^= static_cast<std::uint8_t>((bits >> 8U) & 0xFFU);
        checksum *= kFnvPrime;
    }
    return checksum;
}

}  // namespace

RfFileFacts inspect_rf_file(const RfInputConfig& input) {
    if (input.dtype != "int16_le") {
        throw RfInputError("RF dtype must be int16_le");
    }
    if (input.layout != std::vector<std::string>{
                            "frame", "sequence", "sample", "source_channel"}) {
        throw RfInputError(
            "RF layout must be [frame,sequence,sample,source_channel]");
    }
    if (input.sequences == 0 || input.channels == 0 || input.samples == 0) {
        throw RfInputError(
            "RF sequences, channels, and samples must be positive");
    }
    if (input.frame_count == 0) {
        throw RfInputError("RF frame_count must be positive");
    }
    if (!std::filesystem::exists(input.path)) {
        throw RfInputError(
            "RF input path does not exist: " + input.path.string());
    }
    if (!std::filesystem::is_regular_file(input.path)) {
        throw RfInputError(
            "RF input path is not a regular file: " + input.path.string());
    }
    if constexpr (std::endian::native != std::endian::little) {
        throw RfInputError(
            "int16_le RF input requires a little-endian host in v1");
    }

    std::error_code error;
    const std::uintmax_t file_size =
        std::filesystem::file_size(input.path, error);
    if (error) {
        throw RfInputError(
            "failed to obtain RF input size: " + error.message());
    }
    if (file_size == 0) {
        throw RfInputError("RF input file must not be empty");
    }

    const std::uintmax_t frame_samples =
        static_cast<std::uintmax_t>(samples_per_frame(input));
    const std::uintmax_t frame_bytes =
        bytes_for_samples(frame_samples, "RF frame byte count");
    if (file_size % frame_bytes != 0) {
        throw RfInputError(
            "RF file size " + std::to_string(file_size) +
            " is not an integer multiple of frame_bytes " +
            std::to_string(frame_bytes));
    }
    const std::uintmax_t total_frames_wide = file_size / frame_bytes;
    if (total_frames_wide >
        static_cast<std::uintmax_t>(std::numeric_limits<std::size_t>::max())) {
        throw RfInputError("RF total frame count exceeds size_t");
    }
    const std::size_t total_frames =
        static_cast<std::size_t>(total_frames_wide);
    if (input.frame_start > total_frames ||
        input.frame_count > total_frames - input.frame_start) {
        throw RfInputError(
            "RF frame window start=" + std::to_string(input.frame_start) +
            ", count=" + std::to_string(input.frame_count) +
            " exceeds total_frames " + std::to_string(total_frames));
    }

    return RfFileFacts{
        file_size,
        frame_bytes,
        checked_multiply(
            frame_bytes,
            static_cast<std::uintmax_t>(input.frame_count),
            "selected RF byte count"),
        total_frames};
}

std::vector<std::int16_t> map_source_rf_to_internal(
    const std::vector<std::int16_t>& source_values,
    std::size_t frames,
    std::size_t sequences,
    std::size_t samples,
    std::size_t channels,
    ChannelOrder source_order) {
    std::size_t expected = checked_multiply(
        frames, sequences, "mapped RF source value count");
    expected =
        checked_multiply(expected, samples, "mapped RF source value count");
    expected =
        checked_multiply(expected, channels, "mapped RF source value count");
    if (source_values.size() != expected) {
        throw RfInputError(
            "source RF value count " +
            std::to_string(source_values.size()) +
            " does not match declared four-dimensional shape " +
            std::to_string(expected));
    }

    std::vector<std::int16_t> internal_values(expected);
    for (std::size_t frame = 0; frame < frames; ++frame) {
        for (std::size_t sequence = 0; sequence < sequences; ++sequence) {
            for (std::size_t sample = 0; sample < samples; ++sample) {
                for (std::size_t source_channel = 0;
                     source_channel < channels;
                     ++source_channel) {
                    const std::size_t internal_channel =
                        source_to_internal_channel(
                            source_channel, channels, source_order);
                    const std::size_t source_index = source_rf_index(
                        frame,
                        sequence,
                        sample,
                        source_channel,
                        sequences,
                        samples,
                        channels);
                    const std::size_t internal_index = internal_rf_index(
                        frame,
                        sequence,
                        internal_channel,
                        sample,
                        sequences,
                        channels,
                        samples);
                    internal_values[internal_index] =
                        source_values[source_index];
                }
            }
        }
    }
    return internal_values;
}

MappedRfChunk read_mapped_rf_chunk(
    const RfInputConfig& input,
    const RfFileFacts& facts,
    std::size_t selected_frame_offset,
    std::size_t frame_count) {
    if (frame_count == 0) {
        throw RfInputError("RF chunk frame_count must be positive");
    }
    if (selected_frame_offset > input.frame_count ||
        frame_count > input.frame_count - selected_frame_offset) {
        throw RfInputError(
            "RF chunk is outside the configured selected frame window");
    }

    const std::size_t source_frame_start =
        input.frame_start + selected_frame_offset;
    const std::uintmax_t byte_offset = checked_multiply(
        facts.frame_bytes,
        static_cast<std::uintmax_t>(source_frame_start),
        "RF chunk byte offset");
    const std::uintmax_t byte_count = checked_multiply(
        facts.frame_bytes,
        static_cast<std::uintmax_t>(frame_count),
        "RF chunk byte count");
    const std::uintmax_t value_count_wide =
        byte_count / sizeof(std::int16_t);
    if (value_count_wide >
        static_cast<std::uintmax_t>(std::numeric_limits<std::size_t>::max())) {
        throw RfInputError("RF chunk value count exceeds size_t");
    }

    std::vector<std::int16_t> source_values(
        static_cast<std::size_t>(value_count_wide));
    std::ifstream stream(input.path, std::ios::binary);
    if (!stream) {
        throw RfInputError(
            "failed to open RF input for reading: " + input.path.string());
    }
    stream.seekg(checked_stream_offset(byte_offset), std::ios::beg);
    if (!stream) {
        throw RfInputError(
            "failed to seek RF input to byte offset " +
            std::to_string(byte_offset));
    }
    stream.read(
        reinterpret_cast<char*>(source_values.data()),
        checked_stream_size(byte_count));
    if (stream.gcount() != checked_stream_size(byte_count)) {
        throw RfInputError(
            "short RF read: expected " + std::to_string(byte_count) +
            " bytes, actual " +
            std::to_string(static_cast<std::uintmax_t>(stream.gcount())));
    }

    return MappedRfChunk{
        source_frame_start,
        frame_count,
        input.sequences,
        input.channels,
        input.samples,
        map_source_rf_to_internal(
            source_values,
            frame_count,
            input.sequences,
            input.samples,
            input.channels,
            input.channel_order)};
}

RfWindowValidation validate_and_scan_rf_window(
    const RfInputConfig& input,
    std::size_t chunk_frames) {
    if (chunk_frames == 0) {
        throw RfInputError("RF scan chunk_frames must be positive");
    }
    const RfFileFacts facts = inspect_rf_file(input);
    const bool mapping_bijective =
        channel_mapping_is_bijective(input.channels, input.channel_order);
    if (!mapping_bijective) {
        throw RfInputError(
            "source-to-internal channel mapping is not bijective");
    }

    std::uint64_t checksum = kFnvOffsetBasis;
    std::uintmax_t mapped_sample_count = 0;
    for (std::size_t selected_offset = 0;
         selected_offset < input.frame_count;) {
        const std::size_t current_frames =
            std::min(chunk_frames, input.frame_count - selected_offset);
        const MappedRfChunk chunk = read_mapped_rf_chunk(
            input, facts, selected_offset, current_frames);
        checksum = update_checksum(checksum, chunk.values);
        mapped_sample_count = checked_add(
            mapped_sample_count,
            static_cast<std::uintmax_t>(chunk.values.size()),
            "mapped RF sample count");
        selected_offset += current_frames;
    }

    const std::uintmax_t expected_sample_count =
        facts.selected_bytes / sizeof(std::int16_t);
    if (mapped_sample_count != expected_sample_count) {
        throw RfInputError(
            "mapped RF sample count does not match selected byte count");
    }
    return RfWindowValidation{
        facts,
        checksum,
        mapped_sample_count,
        mapping_bijective,
        mapping_bijective && mapped_sample_count == expected_sample_count};
}

}  // namespace carotid_doppler
