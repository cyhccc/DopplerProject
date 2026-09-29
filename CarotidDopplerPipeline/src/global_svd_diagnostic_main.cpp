#include "carotid_doppler/global_temporal_svd.hpp"

#include <algorithm>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace {

constexpr std::string_view kFilteredIqFile = "filtered_iq_c64.bin";
constexpr std::string_view kPowerFile = "power_linear_f32.bin";
constexpr std::string_view kSingularValuesFile = "singular_values_f32.bin";

struct CliOptions {
    bool show_help{false};
    bool dump_filtered_iq{false};
    std::filesystem::path input_path;
    std::filesystem::path output_directory;
    std::size_t pixels{0};
    std::size_t frames{0};
    std::size_t keep_start{0};
    std::size_t keep_end{0};
};

void print_usage(std::ostream& output) {
    output
        << "Usage:\n"
        << "  carotid_global_svd_diagnostic --input <beamformed_iq_c64.bin> "
           "--output-dir <new-directory> --pixels <count> --frames <count> "
           "--keep-start <1-based> --keep-end <1-based> [--dump-filtered-iq]\n\n"
        << "Input layout is complex float32 [frame,pixel]. The command runs one "
           "global temporal SVD over the complete frame set and writes only "
           "linear float32 diagnostics. It never writes dB arrays or images.\n";
}

std::size_t parse_positive_size(std::string_view text, std::string_view option) {
    std::size_t value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
        value == 0) {
        throw std::invalid_argument(
            std::string(option) + " expects a positive integer, got '" +
            std::string(text) + "'");
    }
    return value;
}

CliOptions parse_cli(int argc, char** argv) {
    CliOptions options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--help" || argument == "-h") {
            options.show_help = true;
        } else if (argument == "--dump-filtered-iq") {
            options.dump_filtered_iq = true;
        } else if (
            argument == "--input" || argument == "--output-dir" ||
            argument == "--pixels" || argument == "--frames" ||
            argument == "--keep-start" || argument == "--keep-end") {
            if (index + 1 >= argc) {
                throw std::invalid_argument(std::string(argument) + " requires a value");
            }
            const std::string_view value(argv[++index]);
            if (argument == "--input") {
                options.input_path = std::filesystem::path(value);
            } else if (argument == "--output-dir") {
                options.output_directory = std::filesystem::path(value);
            } else if (argument == "--pixels") {
                options.pixels = parse_positive_size(value, argument);
            } else if (argument == "--frames") {
                options.frames = parse_positive_size(value, argument);
            } else if (argument == "--keep-start") {
                options.keep_start = parse_positive_size(value, argument);
            } else {
                options.keep_end = parse_positive_size(value, argument);
            }
        } else {
            throw std::invalid_argument("unknown option: " + std::string(argument));
        }
    }

    if (options.show_help) {
        return options;
    }
    if (options.input_path.empty()) {
        throw std::invalid_argument("--input is required");
    }
    if (options.output_directory.empty()) {
        throw std::invalid_argument("--output-dir is required");
    }
    if (options.pixels == 0 || options.frames == 0 || options.keep_start == 0 ||
        options.keep_end == 0) {
        throw std::invalid_argument(
            "--pixels, --frames, --keep-start, and --keep-end are required");
    }
    if (options.keep_start > options.keep_end || options.keep_end > options.frames) {
        throw std::invalid_argument(
            "keep range must satisfy 1 <= keep-start <= keep-end <= frames");
    }
    return options;
}

std::size_t checked_product(
    std::size_t left,
    std::size_t right,
    std::string_view name) {
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::overflow_error(std::string(name) + " overflows size_t");
    }
    return left * right;
}

std::vector<carotid_doppler::Complex32> read_iq(
    const std::filesystem::path& path,
    std::size_t pixels,
    std::size_t frames) {
    if (!std::filesystem::is_regular_file(path)) {
        throw std::invalid_argument("input IQ is not a regular file: " + path.string());
    }
    const std::size_t sample_count = checked_product(pixels, frames, "pixels*frames");
    const std::size_t expected_bytes = checked_product(
        sample_count, sizeof(carotid_doppler::Complex32), "input IQ bytes");
    const std::uintmax_t actual_bytes = std::filesystem::file_size(path);
    if (actual_bytes != expected_bytes) {
        throw std::invalid_argument(
            "input IQ size mismatch: expected " + std::to_string(expected_bytes) +
            " bytes, actual " + std::to_string(actual_bytes) + " bytes");
    }

    std::vector<carotid_doppler::Complex32> values(sample_count);
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open input IQ: " + path.string());
    }
    input.read(
        reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(expected_bytes));
    if (!input) {
        throw std::runtime_error("failed to read complete input IQ: " + path.string());
    }
    if (!std::all_of(values.begin(), values.end(), [](const auto& value) {
            return std::isfinite(value.real) && std::isfinite(value.imag);
        })) {
        throw std::invalid_argument("input IQ contains NaN or Inf");
    }
    return values;
}

template <typename T>
void write_binary_atomic(
    const std::filesystem::path& target,
    const std::vector<T>& values) {
    const std::size_t expected_bytes =
        checked_product(values.size(), sizeof(T), "output artifact bytes");
    std::filesystem::path partial = target;
    partial += ".partial";
    std::ofstream output(partial, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("failed to open partial output: " + partial.string());
    }
    output.write(
        reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(expected_bytes));
    output.flush();
    if (!output) {
        throw std::runtime_error("failed to write output: " + partial.string());
    }
    output.close();
    if (std::filesystem::file_size(partial) != expected_bytes) {
        throw std::runtime_error("output size check failed: " + partial.string());
    }
    std::filesystem::rename(partial, target);
}

void validate_result(
    const carotid_doppler::GlobalTemporalSvdResult& result,
    const CliOptions& options) {
    if (result.singular_values.size() != options.frames ||
        result.power_linear.size() != options.pixels) {
        throw std::runtime_error("global SVD returned an unexpected output shape");
    }
    if (options.dump_filtered_iq &&
        result.filtered_iq.size() !=
            checked_product(options.pixels, options.frames, "filtered IQ count")) {
        throw std::runtime_error("global SVD returned an unexpected filtered-IQ shape");
    }
    if (options.dump_filtered_iq &&
        !std::all_of(result.filtered_iq.begin(), result.filtered_iq.end(), [](const auto& value) {
            return std::isfinite(value.real) && std::isfinite(value.imag);
        })) {
        throw std::runtime_error("global SVD returned invalid filtered IQ");
    }
    if (!std::all_of(
            result.singular_values.begin(),
            result.singular_values.end(),
            [](float value) { return std::isfinite(value) && value >= 0.0F; })) {
        throw std::runtime_error("global SVD returned invalid singular values");
    }
    if (!std::is_sorted(
            result.singular_values.begin(),
            result.singular_values.end(),
            std::greater<float>())) {
        throw std::runtime_error("global SVD singular values are not non-increasing");
    }
    if (!std::all_of(
            result.power_linear.begin(),
            result.power_linear.end(),
            [](float value) { return std::isfinite(value) && value >= 0.0F; })) {
        throw std::runtime_error("global SVD returned invalid linear power");
    }
}

void write_outputs(
    const CliOptions& options,
    const carotid_doppler::GlobalTemporalSvdResult& result) {
    if (std::filesystem::exists(options.output_directory)) {
        throw std::invalid_argument("output directory must be new and immutable");
    }
    if (!std::filesystem::create_directories(options.output_directory)) {
        throw std::runtime_error("failed to create output directory");
    }

    try {
        write_binary_atomic(
            options.output_directory / kSingularValuesFile,
            result.singular_values);
        write_binary_atomic(options.output_directory / kPowerFile, result.power_linear);
        if (options.dump_filtered_iq) {
            write_binary_atomic(
                options.output_directory / kFilteredIqFile,
                result.filtered_iq);
        }
    } catch (...) {
        std::error_code error;
        for (const auto file_name : {kFilteredIqFile, kPowerFile, kSingularValuesFile}) {
            const std::filesystem::path target = options.output_directory / file_name;
            std::filesystem::remove(target, error);
            error.clear();
            std::filesystem::path partial = target;
            partial += ".partial";
            std::filesystem::remove(partial, error);
            error.clear();
        }
        std::filesystem::remove(options.output_directory, error);
        throw;
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const CliOptions options = parse_cli(argc, argv);
        if (options.show_help) {
            print_usage(std::cout);
            return 0;
        }
        if constexpr (std::endian::native != std::endian::little) {
            throw std::runtime_error(
                "global SVD diagnostic requires a little-endian host");
        }

        const auto input = read_iq(options.input_path, options.pixels, options.frames);
        const carotid_doppler::GlobalTemporalSvdConfig config{
            options.pixels,
            options.frames,
            options.keep_start,
            options.keep_end,
        };
        const auto result = carotid_doppler::global_temporal_svd_host(
            input.data(), input.size(), config, options.dump_filtered_iq);
        validate_result(result, options);
        write_outputs(options, result);

        std::cout << "complete global temporal SVD: frames=" << options.frames
                  << ", pixels=" << options.pixels << ", keep=["
                  << options.keep_start << ',' << options.keep_end << "]"
                  << ", kept_components="
                  << (options.keep_end - options.keep_start + 1) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "carotid_global_svd_diagnostic: " << error.what() << '\n';
        return 2;
    }
}
