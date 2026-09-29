#include "carotid_doppler/config.hpp"
#include "carotid_doppler/pipeline_runner.hpp"
#include "carotid_doppler/rf_reader.hpp"
#include "carotid_doppler/run_manifest.hpp"
#include "carotid_doppler/synthetic_generator.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

struct CliOptions {
    bool show_help{false};
    bool synthetic{false};
    bool run_config{false};
    bool dump_beamformed_iq{false};
    bool angle_diagnostics_enabled{false};
    bool dump_angle_beamformed_iq{false};
    bool dump_filtered_iq{false};
    bool batch_frames_explicit{false};
    std::filesystem::path config_path;
    std::filesystem::path run_directory;
    std::size_t batch_frames{4};
    std::size_t nx{0};
    std::size_t nz{0};
    std::size_t frames{0};
};

void print_usage(std::ostream& output) {
    output
        << "Usage:\n"
        << "  carotid_doppler_cuda --synthetic --run-dir <directory> "
           "--nx <count> --nz <count> --frames <count>\n"
        << "  carotid_doppler_cuda --config <pipeline.json>\n"
        << "  carotid_doppler_cuda --config <pipeline.json> --run "
           "[--run-dir <new-directory>] [--batch-frames <count>] "
           "[--dump-beamformed-iq] [--angle-diagnostics] "
           "[--dump-angle-beamformed-iq] "
           "[--dump-filtered-iq]\n\n"
        << "The synthetic mode writes only linear float32 artifacts and "
           "run_manifest.json. It never writes dB arrays or images.\n"
        << "Config mode without --run retains the RF validation-only "
           "interface. --run executes the complete numeric "
           "RF->IQ->DAS->global-SVD->linear-power pipeline. CUDA never writes "
           "dB data or images. Without --run-dir, the formal run creates an "
           "immutable UTC-labelled directory below output.run_root.\n";
}

std::size_t parse_positive_size(std::string_view text, std::string_view option) {
    std::size_t value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size() || value == 0) {
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
        } else if (argument == "--synthetic") {
            options.synthetic = true;
        } else if (argument == "--run") {
            options.run_config = true;
        } else if (argument == "--dump-beamformed-iq") {
            options.dump_beamformed_iq = true;
        } else if (argument == "--angle-diagnostics") {
            options.angle_diagnostics_enabled = true;
        } else if (argument == "--dump-angle-beamformed-iq") {
            options.dump_angle_beamformed_iq = true;
            options.angle_diagnostics_enabled = true;
        } else if (argument == "--dump-filtered-iq") {
            options.dump_filtered_iq = true;
        } else if (argument == "--config" || argument == "--run-dir" ||
                   argument == "--batch-frames" || argument == "--nx" ||
                   argument == "--nz" || argument == "--frames") {
            if (index + 1 >= argc) {
                throw std::invalid_argument(std::string(argument) + " requires a value");
            }
            const std::string_view value(argv[++index]);
            if (argument == "--config") {
                options.config_path = std::filesystem::path(value);
            } else if (argument == "--run-dir") {
                options.run_directory = std::filesystem::path(value);
            } else if (argument == "--batch-frames") {
                options.batch_frames = parse_positive_size(value, argument);
                options.batch_frames_explicit = true;
            } else if (argument == "--nx") {
                options.nx = parse_positive_size(value, argument);
            } else if (argument == "--nz") {
                options.nz = parse_positive_size(value, argument);
            } else {
                options.frames = parse_positive_size(value, argument);
            }
        } else {
            throw std::invalid_argument("unknown option: " + std::string(argument));
        }
    }

    if (options.show_help) {
        return options;
    }
    const bool config_mode = !options.config_path.empty();
    if (options.synthetic == config_mode) {
        throw std::invalid_argument(
            "select exactly one mode: --synthetic or --config <pipeline.json>");
    }
    if (config_mode) {
        if (options.nx != 0 || options.nz != 0 || options.frames != 0) {
            throw std::invalid_argument(
                "--nx, --nz, and --frames are synthetic-only options");
        }
        const bool formal_run =
            options.run_config || !options.run_directory.empty();
        if (!formal_run &&
            (options.batch_frames_explicit || options.dump_beamformed_iq ||
             options.angle_diagnostics_enabled ||
             options.dump_angle_beamformed_iq ||
             options.dump_filtered_iq)) {
            throw std::invalid_argument(
                "--batch-frames and IQ dump flags require --run or --run-dir in config mode");
        }
        return options;
    }
    if (options.run_config || options.batch_frames_explicit ||
        options.dump_beamformed_iq ||
        options.angle_diagnostics_enabled ||
        options.dump_angle_beamformed_iq ||
        options.dump_filtered_iq) {
        throw std::invalid_argument(
            "--run, --batch-frames, and IQ dump flags are config-only options");
    }
    if (options.run_directory.empty()) {
        throw std::invalid_argument("--run-dir is required");
    }
    if (options.nx < 2 || options.nz < 2 || options.frames == 0) {
        throw std::invalid_argument("--nx >= 2, --nz >= 2 and --frames >= 1 are required");
    }
    return options;
}

std::string json_escape(std::string_view value) {
    std::ostringstream escaped;
    for (const char character : value) {
        switch (character) {
        case '\\': escaped << "\\\\"; break;
        case '"': escaped << "\\\""; break;
        case '\n': escaped << "\\n"; break;
        case '\r': escaped << "\\r"; break;
        case '\t': escaped << "\\t"; break;
        default: escaped << character; break;
        }
    }
    return escaped.str();
}

std::string checksum_hex(std::uint64_t checksum) {
    std::ostringstream text;
    text << std::hex << std::setfill('0') << std::setw(16) << checksum;
    return text.str();
}

std::string utc_run_timestamp() {
    const std::time_t now = std::chrono::system_clock::to_time_t(
        std::chrono::system_clock::now());
    std::tm utc{};
#ifdef _WIN32
    if (gmtime_s(&utc, &now) != 0) {
        throw std::runtime_error("failed to convert current time to UTC");
    }
#else
    if (gmtime_r(&now, &utc) == nullptr) {
        throw std::runtime_error("failed to convert current time to UTC");
    }
#endif
    std::ostringstream text;
    text << std::put_time(&utc, "%Y%m%dT%H%M%SZ");
    return text.str();
}

std::string safe_input_label(const std::filesystem::path& input_path) {
    const std::string stem = input_path.stem().string();
    std::string label;
    label.reserve(std::min<std::size_t>(stem.size(), 48));
    bool previous_separator = false;
    for (const unsigned char byte : stem) {
        const bool ascii_alnum =
            (byte >= 'a' && byte <= 'z') ||
            (byte >= 'A' && byte <= 'Z') ||
            (byte >= '0' && byte <= '9');
        if (ascii_alnum || byte == '_') {
            label.push_back(static_cast<char>(byte));
            previous_separator = false;
        } else if (!label.empty() && !previous_separator) {
            label.push_back('-');
            previous_separator = true;
        }
        if (label.size() == 48) {
            break;
        }
    }
    while (!label.empty() && label.back() == '-') {
        label.pop_back();
    }
    return label.empty() ? "input" : label;
}

std::string random_hex_suffix() {
    std::random_device source;
    const std::uint32_t value =
        (static_cast<std::uint32_t>(source()) << 16U) ^
        static_cast<std::uint32_t>(source());
    std::ostringstream text;
    text << std::hex << std::setfill('0') << std::setw(8) << value;
    return text.str();
}

std::filesystem::path make_automatic_run_directory(
    const carotid_doppler::PipelineConfig& config) {
    try {
        const std::filesystem::path run_root =
            std::filesystem::absolute(config.output.run_root)
                .lexically_normal();
        std::error_code error;
        if (std::filesystem::exists(run_root, error)) {
            if (error) {
                throw std::runtime_error(
                    "cannot inspect output.run_root: " + error.message());
            }
            if (!std::filesystem::is_directory(run_root, error) || error) {
                throw std::runtime_error(
                    "output.run_root is not a directory: " +
                    run_root.string());
            }
        } else {
            if (error) {
                throw std::runtime_error(
                    "cannot inspect output.run_root: " + error.message());
            }
            if (!std::filesystem::create_directories(run_root, error) || error) {
                throw std::runtime_error(
                    "cannot create output.run_root '" + run_root.string() +
                    "': " + error.message());
            }
        }

        const std::string prefix =
            utc_run_timestamp() + "-" + safe_input_label(config.input.path);
        for (int attempt = 0; attempt < 64; ++attempt) {
            const std::filesystem::path candidate =
                run_root / (prefix + "-" + random_hex_suffix());
            if (!std::filesystem::exists(candidate, error)) {
                if (error) {
                    throw std::runtime_error(
                        "cannot inspect automatic run path: " +
                        error.message());
                }
                return candidate;
            }
            if (error) {
                throw std::runtime_error(
                    "cannot inspect automatic run path: " +
                    error.message());
            }
        }
        throw std::runtime_error(
            "could not find an unused automatic run directory after 64 attempts");
    } catch (const std::exception& error) {
        throw std::runtime_error(
            "failed to generate automatic run directory: " +
            std::string(error.what()));
    }
}

void print_rf_validation_facts(
    std::ostream& output,
    const carotid_doppler::PipelineConfig& config,
    const carotid_doppler::RfWindowValidation& validation) {
    const auto source_order =
        carotid_doppler::channel_order_name(config.input.channel_order);
    const bool reversed =
        config.input.channel_order ==
        carotid_doppler::ChannelOrder::descending_x;
    output
        << "{\n"
        << "  \"schema_version\": 1,\n"
        << "  \"status\": \"rf_window_validated\",\n"
        << "  \"input\": {\n"
        << "    \"path\": \""
        << json_escape(config.input.path.generic_string()) << "\",\n"
        << "    \"dtype\": \"int16_le\",\n"
        << "    \"layout\": [\"frame\", \"sequence\", \"sample\", "
           "\"source_channel\"],\n"
        << "    \"shape\": [" << validation.file.total_frames << ", "
        << config.input.sequences << ", " << config.input.samples << ", "
        << config.input.channels << "],\n"
        << "    \"frame_start\": " << config.input.frame_start << ",\n"
        << "    \"frame_count\": " << config.input.frame_count << ",\n"
        << "    \"size_bytes\": " << validation.file.size_bytes << ",\n"
        << "    \"frame_bytes\": " << validation.file.frame_bytes << ",\n"
        << "    \"selected_bytes\": " << validation.file.selected_bytes
        << "\n"
        << "  },\n"
        << "  \"channel_mapping\": {\n"
        << "    \"source_order\": \"" << source_order << "\",\n"
        << "    \"internal_order\": \"ascending_x\",\n"
        << "    \"reversed\": " << (reversed ? "true" : "false") << ",\n"
        << "    \"formula\": \""
        << (reversed ? "channels - 1 - source_channel" : "source_channel")
        << "\"\n"
        << "  },\n"
        << "  \"validation\": {\n"
        << "    \"internal_layout\": [\"frame\", \"sequence\", "
           "\"internal_channel\", \"sample\"],\n"
        << "    \"mapped_sample_count\": "
        << validation.mapped_sample_count << ",\n"
        << "    \"channel_mapping_bijective\": "
        << (validation.channel_mapping_bijective ? "true" : "false") << ",\n"
        << "    \"sample_multiset_preserved\": "
        << (validation.sample_multiset_preserved ? "true" : "false") << ",\n"
        << "    \"mapped_checksum_fnv1a64\": \""
        << checksum_hex(validation.mapped_checksum_fnv1a64) << "\"\n"
        << "  }\n"
        << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::filesystem::path run_directory;
    try {
        const CliOptions options = parse_cli(argc, argv);
        if (options.show_help) {
            print_usage(std::cout);
            return 0;
        }

        if (!options.config_path.empty()) {
            const bool formal_run =
                options.run_config || !options.run_directory.empty();
            if (formal_run) {
                const auto config =
                    carotid_doppler::load_pipeline_config(options.config_path);
                carotid_doppler::validate_real_carotid_protocol(config);
                std::filesystem::path failed_run_directory;
                bool directory_existed_before = false;
                if (!options.run_directory.empty()) {
                    failed_run_directory =
                        std::filesystem::absolute(options.run_directory)
                            .lexically_normal();
                    directory_existed_before =
                        std::filesystem::exists(failed_run_directory);
                }
                try {
                    if (failed_run_directory.empty()) {
                        failed_run_directory =
                            make_automatic_run_directory(config);
                        directory_existed_before = false;
                    }
                    carotid_doppler::run_real_pipeline(
                        config,
                        carotid_doppler::RealPipelineOptions{
                            failed_run_directory,
                            options.batch_frames,
                            options.dump_beamformed_iq ||
                                config.output.dump_beamformed_iq,
                            options.angle_diagnostics_enabled,
                            options.dump_angle_beamformed_iq,
                            options.dump_filtered_iq ||
                                config.output.dump_filtered_iq});
                } catch (const std::exception& error) {
                    if (!failed_run_directory.empty() &&
                        !directory_existed_before) {
                        try {
                            carotid_doppler::write_failed_real_run_log(
                                failed_run_directory, error.what());
                        } catch (const std::exception& log_error) {
                            std::cerr
                                << "carotid_doppler_cuda: failed to persist "
                                   "run_error.txt: "
                                << log_error.what() << '\n';
                        }
                    }
                    throw;
                }
                std::cout << "complete real numeric run directory: "
                          << failed_run_directory.string() << '\n';
                return 0;
            }
            const auto config =
                carotid_doppler::load_pipeline_config(options.config_path);
            const auto validation =
                carotid_doppler::validate_and_scan_rf_window(config.input);
            print_rf_validation_facts(std::cout, config, validation);
            return 0;
        }

        run_directory = std::filesystem::absolute(options.run_directory).lexically_normal();
        const auto artifacts = carotid_doppler::generate_synthetic_artifacts(
            options.nx, options.nz, options.frames);
        std::string run_id = run_directory.filename().string();
        if (run_id.empty()) {
            run_id = "synthetic-run";
        }
        carotid_doppler::write_complete_synthetic_run(
            carotid_doppler::SyntheticRunSpec{
                run_directory,
                run_id,
                options.nx,
                options.nz,
                options.frames,
            },
            artifacts);
        std::cout << "complete run directory: " << run_directory.string() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "carotid_doppler_cuda: " << error.what() << '\n';
        return 2;
    }
}
