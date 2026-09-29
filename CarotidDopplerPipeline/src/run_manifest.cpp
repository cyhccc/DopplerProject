#include "carotid_doppler/run_manifest.hpp"
#include "carotid_doppler/rf_bandpass.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <vector>

#ifndef CAROTID_BUILD_TYPE
#define CAROTID_BUILD_TYPE "unknown"
#endif

#ifndef CAROTID_GIT_COMMIT
#define CAROTID_GIT_COMMIT "not-recorded"
#endif

#ifndef CAROTID_PROJECT_VERSION
#define CAROTID_PROJECT_VERSION "unknown"
#endif

namespace carotid_doppler {
namespace {

constexpr std::string_view kPowerFile = "power_linear_f32.bin";
constexpr std::string_view kSingularValuesFile = "singular_values_f32.bin";
constexpr std::string_view kBeamformedIqFile = "beamformed_iq_c64.bin";
constexpr std::string_view kAngleBeamformedIqFile =
    "angle_beamformed_iq_c64.bin";
constexpr std::string_view kAnglePowerFile = "angle_power_linear_f32.bin";
constexpr std::string_view kAngleSingularValuesFile =
    "angle_singular_values_f32.bin";
constexpr std::string_view kIncoherentPowerFile =
    "incoherent_power_linear_f32.bin";
constexpr std::string_view kAngleCoherenceFile = "angle_coherence_f32.bin";
constexpr std::string_view kFilteredIqFile = "filtered_iq_c64.bin";
constexpr std::string_view kManifestFile = "run_manifest.json";

template <typename Value>
std::uintmax_t expected_bytes_for(std::size_t value_count) {
    if (value_count >
        std::numeric_limits<std::uintmax_t>::max() / sizeof(Value)) {
        throw std::overflow_error("artifact byte count overflows uintmax_t");
    }
    return static_cast<std::uintmax_t>(value_count) * sizeof(Value);
}

std::uintmax_t expected_bytes(std::size_t value_count) {
    return expected_bytes_for<float>(value_count);
}

void validate_artifacts(
    const SyntheticRunSpec& spec,
    const SyntheticArtifacts& artifacts) {
    if (spec.nx > std::numeric_limits<std::size_t>::max() / spec.nz) {
        throw std::invalid_argument("run nx*nz overflows size_t");
    }
    const std::size_t expected_power_count = spec.nx * spec.nz;
    if (artifacts.power.size() != expected_power_count) {
        throw std::invalid_argument("synthetic power value count does not match [nz,nx]");
    }
    if (artifacts.singular_values.size() != spec.frames) {
        throw std::invalid_argument("synthetic singular value count does not match frames");
    }
    if (!std::all_of(artifacts.power.begin(), artifacts.power.end(), [](float value) {
            return std::isfinite(value) && value >= 0.0F;
        })) {
        throw std::invalid_argument("synthetic power must be finite and non-negative");
    }
    if (!std::all_of(
            artifacts.singular_values.begin(),
            artifacts.singular_values.end(),
            [](float value) { return std::isfinite(value) && value >= 0.0F; })) {
        throw std::invalid_argument("synthetic singular values must be finite and non-negative");
    }
    if (!std::is_sorted(
            artifacts.singular_values.begin(),
            artifacts.singular_values.end(),
            std::greater<float>())) {
        throw std::invalid_argument("synthetic singular values must be non-increasing");
    }
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

void prepare_run_directory(const std::filesystem::path& run_directory) {
    if (std::filesystem::exists(run_directory)) {
        throw std::invalid_argument("run directory must be new and immutable");
    }
    if (!std::filesystem::create_directories(run_directory)) {
        throw std::runtime_error("failed to create run directory");
    }
}

template <typename Value>
void write_binary_atomic(
    const std::filesystem::path& target,
    const std::vector<Value>& values) {
    std::filesystem::path partial = target;
    partial += ".partial";
    std::ofstream output(partial, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("failed to open partial binary artifact: " + partial.string());
    }
    output.write(
        reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(expected_bytes_for<Value>(values.size())));
    output.flush();
    if (!output) {
        throw std::runtime_error("failed to write binary artifact: " + partial.string());
    }
    output.close();
    if (std::filesystem::file_size(partial) !=
        expected_bytes_for<Value>(values.size())) {
        throw std::runtime_error("binary artifact size check failed: " + partial.string());
    }
    std::filesystem::rename(partial, target);
}

void write_text_atomic(const std::filesystem::path& target, std::string_view text) {
    std::filesystem::path partial = target;
    partial += ".partial";
    std::ofstream output(partial, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("failed to open partial manifest: " + partial.string());
    }
    output.write(text.data(), static_cast<std::streamsize>(text.size()));
    output.flush();
    if (!output) {
        throw std::runtime_error("failed to write manifest: " + partial.string());
    }
    output.close();
    std::filesystem::rename(partial, target);
}

std::string build_manifest(
    const SyntheticRunSpec& spec,
    const SyntheticArtifacts& artifacts) {
    const auto power_bytes = expected_bytes(artifacts.power.size());
    const auto singular_bytes = expected_bytes(artifacts.singular_values.size());
    const double dx_m = 0.04 / static_cast<double>(spec.nx - 1);
    const double dz_m = 0.07 / static_cast<double>(spec.nz - 1);

    std::ostringstream json;
    json << std::setprecision(17);
    json << "{\n"
         << "  \"schema_version\": 1,\n"
         << "  \"run_id\": \"" << json_escape(spec.run_id) << "\",\n"
         << "  \"status\": \"complete\",\n"
         << "  \"input\": {\n"
         << "    \"path\": \"synthetic://ticket-1/deterministic-power-v1\",\n"
         << "    \"dtype\": \"float32_le\",\n"
         << "    \"layout\": [\"frame\", \"z\", \"x\"],\n"
         << "    \"shape\": [" << spec.frames << ", " << spec.nz << ", " << spec.nx << "],\n"
         << "    \"frame_start\": 0,\n"
         << "    \"frame_count\": " << spec.frames << ",\n"
         << "    \"fingerprint\": {\"size_bytes\": 0, \"last_write_time_utc\": \"1970-01-01T00:00:00Z\", \"sha256\": null}\n"
         << "  },\n"
         << "  \"channel_mapping\": {\"source_order\": \"ascending_x\", \"internal_order\": \"ascending_x\", \"reversed\": false},\n"
         << "  \"acquisition\": {\"sample_rate_hz\": 1, \"center_frequency_hz\": 1, \"sound_speed_m_per_s\": 1, \"pitch_m\": 1, \"transmit_angles_deg\": [0], \"tx_delay_offset_samples\": 0, \"tx_reference_mode\": \"aligned_reference\"},\n"
         << "  \"image_grid\": {\n"
         << "    \"nx\": " << spec.nx << ", \"nz\": " << spec.nz << ",\n"
         << "    \"x_min_m\": -0.02, \"x_max_m\": 0.02,\n"
         << "    \"z_min_m\": 0.0, \"z_max_m\": 0.07,\n"
         << "    \"dx_m\": " << dx_m << ", \"dz_m\": " << dz_m << ",\n"
         << "    \"layout\": [\"z\", \"x\"]\n"
         << "  },\n"
         << "  \"processing\": {\n"
         << "    \"pipeline_mode\": \"synthetic_contract\",\n"
         << "    \"bandpass_enabled\": false, \"global_temporal_svd\": false,\n"
         << "    \"svd_keep_start\": null, \"svd_keep_end\": null,\n"
         << "    \"svd_indexing\": \"one_based_inclusive\",\n"
         << "    \"motion_correction\": false, \"block_svd\": false, \"channel_repair\": false\n"
         << "  },\n"
         << "  \"artifacts\": {\n"
         << "    \"power_linear\": {\"path\": \"" << kPowerFile
         << "\", \"dtype\": \"float32_le\", \"shape\": [" << spec.nz << ", " << spec.nx
         << "], \"layout\": [\"z\", \"x\"], \"size_bytes\": " << power_bytes << "},\n"
         << "    \"singular_values\": {\"path\": \"" << kSingularValuesFile
         << "\", \"dtype\": \"float32_le\", \"shape\": [" << spec.frames
         << "], \"layout\": [\"component\"], \"size_bytes\": " << singular_bytes << "},\n"
         << "    \"beamformed_iq\": null, "
            "\"angle_beamformed_iq\": null, "
            "\"angle_power_linear\": null, "
            "\"angle_singular_values\": null, "
            "\"incoherent_power_linear\": null, "
            "\"angle_coherence\": null, \"filtered_iq\": null\n"
         << "  },\n"
         << "  \"software\": {\"git_commit\": \"not-recorded\", \"build_type\": \""
         << json_escape(CAROTID_BUILD_TYPE) << "\", \"cuda_runtime\": \""
         << json_escape(artifacts.cuda_runtime) << "\", \"gpu_name\": \""
         << json_escape(artifacts.gpu_name) << "\"},\n"
         << "  \"timings_ms\": {\"read_and_map\": 0, \"bandpass\": 0, \"rf_to_iq\": 0, \"beamforming\": 0, \"covariance\": 0, \"svd\": 0, \"projection\": 0, \"power\": 0, \"write\": 0, \"total\": 0}\n"
         << "}\n";
    return json.str();
}

std::size_t checked_product(
    std::size_t left,
    std::size_t right,
    std::string_view description) {
    if (left != 0 &&
        right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::overflow_error(
            std::string(description) + " overflows size_t");
    }
    return left * right;
}

bool finite_complex(const Complex32& value) {
    return std::isfinite(value.real) && std::isfinite(value.imag);
}

void validate_real_artifacts(
    const RealRunSpec& spec,
    const RealRunArtifacts& artifacts) {
    if (spec.batch_frames == 0) {
        throw std::invalid_argument("real pipeline batch_frames must be positive");
    }
    if (spec.dump_angle_beamformed_iq &&
        !spec.angle_diagnostics_enabled) {
        throw std::invalid_argument(
            "dump_angle_beamformed_iq requires angle diagnostics");
    }
    const BandpassFileIdentity& bandpass = spec.bandpass_identity;
    if (bandpass.enabled != spec.config.bandpass.enabled ||
        bandpass.dtype != spec.config.bandpass.dtype ||
        bandpass.taps != spec.config.bandpass.taps ||
        !bandpass.coefficients_path.is_absolute()) {
        throw std::invalid_argument(
            "captured FIR identity does not match the normalized configuration");
    }
    if (bandpass.enabled &&
        (!bandpass.fingerprint_present || bandpass.size_bytes == 0 ||
         bandpass.last_write_time_utc.empty())) {
        throw std::invalid_argument(
            "enabled FIR identity requires a complete file fingerprint");
    }
    if (!bandpass.enabled && bandpass.fingerprint_present) {
        throw std::invalid_argument(
            "disabled FIR identity must not contain a file fingerprint");
    }
    const std::size_t pixels = checked_product(
        spec.config.image_grid.nx,
        spec.config.image_grid.nz,
        "real pipeline pixel count");
    const std::size_t iq_values = checked_product(
        pixels,
        spec.config.input.frame_count,
        "real pipeline IQ value count");
    const std::size_t angle_iq_values = checked_product(
        iq_values,
        spec.config.input.sequences,
        "real pipeline per-angle IQ value count");
    if (artifacts.power_linear.size() != pixels) {
        throw std::invalid_argument(
            "real power value count does not match [nz,nx]");
    }
    if (artifacts.singular_values.size() !=
        spec.config.input.frame_count) {
        throw std::invalid_argument(
            "real singular-value count does not match frame_count");
    }
    if (spec.dump_beamformed_iq) {
        if (artifacts.beamformed_iq.size() != iq_values) {
            throw std::invalid_argument(
                "beamformed IQ value count does not match [frame,nz,nx]");
        }
    } else if (!artifacts.beamformed_iq.empty()) {
        throw std::invalid_argument(
            "beamformed IQ was supplied while its diagnostic dump is disabled");
    }
    if (spec.dump_angle_beamformed_iq) {
        if (artifacts.angle_beamformed_iq.size() != angle_iq_values) {
            throw std::invalid_argument(
                "per-angle IQ value count does not match [frame,angle,nz,nx]");
        }
    } else if (!artifacts.angle_beamformed_iq.empty()) {
        throw std::invalid_argument(
            "angle-beamformed IQ was supplied while its dump is disabled");
    }
    if (spec.angle_diagnostics_enabled) {
        if (artifacts.angle_power_linear.size() !=
            spec.config.input.sequences * pixels) {
            throw std::invalid_argument(
                "angle power value count does not match [angle,nz,nx]");
        }
        if (artifacts.angle_singular_values.size() !=
            spec.config.input.sequences * spec.config.input.frame_count) {
            throw std::invalid_argument(
                "angle singular-value count does not match [angle,component]");
        }
        if (artifacts.incoherent_power_linear.size() != pixels ||
            artifacts.angle_coherence.size() != pixels) {
            throw std::invalid_argument(
                "incoherent power and angle coherence must match [nz,nx]");
        }
    } else if (!artifacts.angle_power_linear.empty() ||
               !artifacts.angle_singular_values.empty() ||
               !artifacts.incoherent_power_linear.empty() ||
               !artifacts.angle_coherence.empty()) {
        throw std::invalid_argument(
            "angle diagnostics were supplied while their dump is disabled");
    }
    if (spec.dump_filtered_iq) {
        if (artifacts.filtered_iq.size() != iq_values) {
            throw std::invalid_argument(
                "filtered IQ value count does not match [frame,nz,nx]");
        }
    } else if (!artifacts.filtered_iq.empty()) {
        throw std::invalid_argument(
            "filtered IQ was supplied while its diagnostic dump is disabled");
    }
    if (!std::all_of(
            artifacts.power_linear.begin(),
            artifacts.power_linear.end(),
            [](float value) { return std::isfinite(value) && value >= 0.0F; })) {
        throw std::invalid_argument(
            "real linear power must be finite and non-negative");
    }
    if (!std::all_of(
            artifacts.singular_values.begin(),
            artifacts.singular_values.end(),
            [](float value) { return std::isfinite(value) && value >= 0.0F; })) {
        throw std::invalid_argument(
            "real singular values must be finite and non-negative");
    }
    if (!std::is_sorted(
            artifacts.singular_values.begin(),
            artifacts.singular_values.end(),
            std::greater<float>())) {
        throw std::invalid_argument(
            "real singular values must be non-increasing");
    }
    if (spec.dump_beamformed_iq &&
        !std::all_of(
            artifacts.beamformed_iq.begin(),
            artifacts.beamformed_iq.end(),
            finite_complex)) {
        throw std::invalid_argument("beamformed IQ contains NaN or Inf");
    }
    if (spec.dump_filtered_iq &&
        !std::all_of(
            artifacts.filtered_iq.begin(),
            artifacts.filtered_iq.end(),
            finite_complex)) {
        throw std::invalid_argument("filtered IQ contains NaN or Inf");
    }
    if (spec.dump_angle_beamformed_iq &&
        !std::all_of(
            artifacts.angle_beamformed_iq.begin(),
            artifacts.angle_beamformed_iq.end(),
            finite_complex)) {
        throw std::invalid_argument("per-angle IQ contains NaN or Inf");
    }
    if (spec.angle_diagnostics_enabled &&
        !std::all_of(
            artifacts.angle_power_linear.begin(),
            artifacts.angle_power_linear.end(),
            [](float value) { return std::isfinite(value) && value >= 0.0F; })) {
        throw std::invalid_argument(
            "angle linear power contains invalid values");
    }
    if (spec.angle_diagnostics_enabled &&
        !std::all_of(
            artifacts.angle_singular_values.begin(),
            artifacts.angle_singular_values.end(),
            [](float value) { return std::isfinite(value) && value >= 0.0F; })) {
        throw std::invalid_argument(
            "angle singular values contain invalid values");
    }
    if (spec.angle_diagnostics_enabled &&
        !std::all_of(
            artifacts.incoherent_power_linear.begin(),
            artifacts.incoherent_power_linear.end(),
            [](float value) { return std::isfinite(value) && value >= 0.0F; })) {
        throw std::invalid_argument(
            "incoherent linear power contains invalid values");
    }
    if (spec.angle_diagnostics_enabled &&
        !std::all_of(
            artifacts.angle_coherence.begin(),
            artifacts.angle_coherence.end(),
            [](float value) {
                return std::isfinite(value) && value >= 0.0F &&
                       value <= 1.00001F;
            })) {
        throw std::invalid_argument("angle coherence is outside [0,1]");
    }
}

std::string file_last_write_time_utc(const std::filesystem::path& path) {
    std::error_code error;
    const auto file_time = std::filesystem::last_write_time(path, error);
    if (error) {
        throw std::runtime_error(
            "failed to read file last-write time: " + error.message());
    }
    const auto system_time = std::chrono::time_point_cast<
        std::chrono::system_clock::duration>(
        file_time - std::filesystem::file_time_type::clock::now() +
        std::chrono::system_clock::now());
    const std::time_t raw_time =
        std::chrono::system_clock::to_time_t(system_time);
    std::tm utc{};
#ifdef _WIN32
    if (gmtime_s(&utc, &raw_time) != 0) {
        throw std::runtime_error("failed to convert RF last-write time to UTC");
    }
#else
    if (gmtime_r(&raw_time, &utc) == nullptr) {
        throw std::runtime_error("failed to convert RF last-write time to UTC");
    }
#endif
    std::ostringstream output;
    output << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    return output.str();
}

std::filesystem::path normalized_bandpass_path(
    const BandpassConfig& bandpass) {
    if (!bandpass.enabled) {
        return std::filesystem::absolute(bandpass.coefficients_path)
            .lexically_normal();
    }
    std::error_code error;
    const std::filesystem::path canonical_path =
        std::filesystem::canonical(bandpass.coefficients_path, error);
    if (error) {
        throw std::runtime_error(
            "failed to canonicalize enabled FIR coefficient path: " +
            error.message());
    }
    return canonical_path;
}

void append_bandpass_identity(
    std::ostringstream& json,
    const BandpassFileIdentity& bandpass) {
    json << "    \"bandpass\": {\n"
         << "      \"enabled\": "
         << (bandpass.enabled ? "true" : "false") << ",\n"
         << "      \"coefficients_path\": \""
         << json_escape(bandpass.coefficients_path.generic_string())
         << "\",\n"
         << "      \"dtype\": \"" << json_escape(bandpass.dtype)
         << "\",\n"
         << "      \"taps\": " << bandpass.taps << ",\n"
         << "      \"fingerprint\": ";
    if (!bandpass.fingerprint_present) {
        json << "null,\n"
             << "      \"disabled_semantics\": "
                "\"coefficient file was not opened or fingerprinted\"\n";
    } else {
        json << "{\"size_bytes\": " << bandpass.size_bytes
             << ", \"last_write_time_utc\": \""
             << json_escape(bandpass.last_write_time_utc)
             << "\", \"sha256\": null},\n"
             << "      \"disabled_semantics\": null\n";
    }
    json << "    },\n";
}

void append_iq_artifact(
    std::ostringstream& json,
    std::string_view key,
    std::string_view file_name,
    bool enabled,
    std::size_t frames,
    std::size_t nz,
    std::size_t nx) {
    json << "    \"" << key << "\": ";
    if (!enabled) {
        json << "null";
        return;
    }
    const std::size_t value_count = checked_product(
        checked_product(frames, nz, "IQ artifact value count"),
        nx,
        "IQ artifact value count");
    json << "{\"path\": \"" << file_name
         << "\", \"scalar_dtype\": \"float32_le\", "
            "\"complex_encoding\": \"interleaved_real_imag\", "
            "\"shape\": ["
         << frames << ", " << nz << ", " << nx
         << "], \"layout\": [\"frame\", \"z\", \"x\"], "
            "\"size_bytes\": "
         << expected_bytes_for<Complex32>(value_count) << "}";
}

void append_angle_diagnostic_artifacts(
    std::ostringstream& json,
    bool diagnostics_enabled,
    bool dump_iq,
    std::size_t frames,
    std::size_t angles,
    std::size_t nz,
    std::size_t nx) {
    json << "    \"angle_beamformed_iq\": ";
    if (!dump_iq) {
        json << "null,\n";
    } else {
        const std::size_t value_count = checked_product(
            checked_product(
                checked_product(
                    frames, angles, "per-angle IQ artifact value count"),
                nz,
                "per-angle IQ artifact value count"),
            nx,
            "per-angle IQ artifact value count");
        json << "{\"path\": \"" << kAngleBeamformedIqFile
             << "\", \"scalar_dtype\": \"float32_le\", "
                "\"complex_encoding\": \"interleaved_real_imag\", "
                "\"shape\": ["
             << frames << ", " << angles << ", " << nz << ", " << nx
             << "], \"layout\": [\"frame\", \"angle\", \"z\", \"x\"], "
                "\"size_bytes\": "
             << expected_bytes_for<Complex32>(value_count) << "},\n";
    }
    if (!diagnostics_enabled) {
        json << "    \"angle_power_linear\": null,\n"
             << "    \"angle_singular_values\": null,\n"
             << "    \"incoherent_power_linear\": null,\n"
             << "    \"angle_coherence\": null";
        return;
    }
    const std::size_t power_count = checked_product(
        checked_product(angles, nz, "angle power artifact value count"),
        nx,
        "angle power artifact value count");
    json << "    \"angle_power_linear\": {\"path\": \""
         << kAnglePowerFile
         << "\", \"dtype\": \"float32_le\", \"shape\": ["
         << angles << ", " << nz << ", " << nx
         << "], \"layout\": [\"angle\", \"z\", \"x\"], \"size_bytes\": "
         << expected_bytes_for<float>(power_count) << "},\n"
         << "    \"angle_singular_values\": {\"path\": \""
         << kAngleSingularValuesFile
         << "\", \"dtype\": \"float32_le\", \"shape\": ["
         << angles << ", " << frames
         << "], \"layout\": [\"angle\", \"component\"], \"size_bytes\": "
         << expected_bytes_for<float>(
                checked_product(angles, frames, "angle singular artifact count"))
         << "},\n"
         << "    \"incoherent_power_linear\": {\"path\": \""
         << kIncoherentPowerFile
         << "\", \"dtype\": \"float32_le\", \"shape\": ["
         << nz << ", " << nx
         << "], \"layout\": [\"z\", \"x\"], \"size_bytes\": "
         << expected_bytes_for<float>(
                checked_product(nz, nx, "incoherent power artifact count"))
         << "},\n"
         << "    \"angle_coherence\": {\"path\": \""
         << kAngleCoherenceFile
         << "\", \"dtype\": \"float32_le\", \"shape\": ["
         << nz << ", " << nx
         << "], \"layout\": [\"z\", \"x\"], \"size_bytes\": "
         << expected_bytes_for<float>(
                checked_product(nz, nx, "angle coherence artifact count"))
         << "}";
}

std::string build_real_manifest(
    const RealRunSpec& spec,
    const RealRunArtifacts& artifacts,
    const PipelineTimings& timings) {
    const PipelineConfig& config = spec.config;
    const std::size_t pixels = checked_product(
        config.image_grid.nx,
        config.image_grid.nz,
        "manifest pixel count");
    const bool reversed =
        config.input.channel_order == ChannelOrder::descending_x;
    const RfTgcParameters tgc_parameters =
        make_rf_tgc_parameters(config);
    const double dx_m =
        (config.image_grid.x_max_m - config.image_grid.x_min_m) /
        static_cast<double>(config.image_grid.nx - 1);
    const double dz_m =
        (config.image_grid.z_max_m - config.image_grid.z_min_m) /
        static_cast<double>(config.image_grid.nz - 1);

    std::ostringstream json;
    json << std::setprecision(17)
         << "{\n"
         << "  \"schema_version\": 1,\n"
         << "  \"run_id\": \"" << json_escape(spec.run_id) << "\",\n"
         << "  \"status\": \"complete\",\n"
         << "  \"config_path\": \""
         << json_escape(config.source_path.generic_string()) << "\",\n"
         << "  \"input\": {\n"
         << "    \"path\": \""
         << json_escape(config.input.path.generic_string()) << "\",\n"
         << "    \"dtype\": \"int16_le\",\n"
         << "    \"layout\": [\"frame\", \"sequence\", \"sample\", \"source_channel\"],\n"
         << "    \"shape\": [" << spec.input_facts.total_frames << ", "
         << config.input.sequences << ", " << config.input.samples << ", "
         << config.input.channels << "],\n"
         << "    \"frame_start\": " << config.input.frame_start << ",\n"
         << "    \"frame_count\": " << config.input.frame_count << ",\n"
         << "    \"frame_end_exclusive\": "
         << config.input.frame_start + config.input.frame_count << ",\n"
         << "    \"frame_bytes\": " << spec.input_facts.frame_bytes << ",\n"
         << "    \"selected_bytes\": " << spec.input_facts.selected_bytes
         << ",\n"
         << "    \"fingerprint\": {\"size_bytes\": "
         << spec.input_facts.size_bytes << ", \"last_write_time_utc\": \""
         << file_last_write_time_utc(config.input.path)
         << "\", \"sha256\": null}\n"
         << "  },\n"
         << "  \"channel_mapping\": {\"source_order\": \""
         << channel_order_name(config.input.channel_order)
         << "\", \"internal_order\": \"ascending_x\", \"reversed\": "
         << (reversed ? "true" : "false")
         << ", \"formula\": \""
         << (reversed ? "channels - 1 - source_channel" : "source_channel")
         << "\"},\n"
         << "  \"acquisition\": {\n"
         << "    \"sample_rate_hz\": "
         << config.acquisition.sample_rate_hz << ",\n"
         << "    \"center_frequency_hz\": "
         << config.acquisition.center_frequency_hz << ",\n"
         << "    \"sound_speed_m_per_s\": "
         << config.acquisition.sound_speed_m_per_s << ",\n"
         << "    \"pitch_m\": " << config.acquisition.pitch_m << ",\n"
         << "    \"transmit_angles_deg\": [";
    for (std::size_t index = 0;
         index < config.acquisition.transmit_angles_deg.size(); ++index) {
        if (index != 0) {
            json << ", ";
        }
        json << config.acquisition.transmit_angles_deg[index];
    }
    json << "],\n"
         << "    \"tx_delay_offset_samples\": "
         << config.acquisition.tx_delay_offset_samples << ",\n"
         << "    \"transmit_cycle_count\": "
         << config.acquisition.transmit_cycle_count << ",\n"
         << "    \"tx_reference_mode\": \""
         << json_escape(config.acquisition.tx_reference_mode) << "\"\n"
         << "  },\n"
         << "  \"receive\": {\"f_number\": " << config.receive.f_number
         << ", \"minimum_aperture_m\": "
         << config.receive.minimum_aperture_m
         << ", \"apodization\": \""
         << json_escape(config.receive.apodization) << "\"},\n"
         << "  \"image_grid\": {\n"
         << "    \"nx\": " << config.image_grid.nx
         << ", \"nz\": " << config.image_grid.nz << ",\n"
         << "    \"x_min_m\": " << config.image_grid.x_min_m
         << ", \"x_max_m\": " << config.image_grid.x_max_m << ",\n"
         << "    \"z_min_m\": " << config.image_grid.z_min_m
         << ", \"z_max_m\": " << config.image_grid.z_max_m << ",\n"
         << "    \"dx_m\": " << dx_m << ", \"dz_m\": " << dz_m
         << ", \"layout\": [\"z\", \"x\"]\n"
         << "  },\n"
         << "  \"processing\": {\n"
         << "    \"pipeline_mode\": \"real_carotid_numeric\",\n"
         << "    \"rf_batch_frames\": " << spec.batch_frames << ",\n"
         << "    \"internal_rf_layout\": [\"frame\", \"sequence\", \"internal_channel\", \"sample\"],\n";
    json << "    \"tgc\": {\"enabled\": "
         << (config.tgc.enabled ? "true" : "false")
         << ", \"alpha_coeff_db_cm_mhz_power\": "
         << config.tgc.alpha_coeff_db_cm_mhz_power
         << ", \"frequency_power\": " << config.tgc.frequency_power
         << ", \"scale_per_sample\": "
         << tgc_parameters.scale_per_sample
         << ", \"offset\": " << tgc_parameters.offset
         << ", \"formula\": \"gain(sample0based)=exp((sample+1)*scale+offset)\", \"stage\": \"after_int16_mapping_before_centered_bandpass\"},\n";
    append_bandpass_identity(json, spec.bandpass_identity);
    json << "    \"iq_method\": \"fft_discrete_hilbert\",\n"
         << "    \"beamformed_iq_layout\": [\"frame\", \"z\", \"x\"],\n"
         << "    \"angle_beamformed_iq_stage\": \"pre_svd\",\n"
         << "    \"angle_diagnostics_enabled\": "
         << (spec.angle_diagnostics_enabled ? "true" : "false") << ",\n"
         << "    \"dump_angle_beamformed_iq\": "
         << (spec.dump_angle_beamformed_iq ? "true" : "false") << ",\n"
         << "    \"angle_power_svd_band\": [" << config.svd.keep_start
         << ", " << config.svd.keep_end << "],\n"
         << "    \"power_linear_semantics\": \"coherent angle sum followed by one global temporal SVD\",\n"
         << "    \"coherent_power_linear_artifact\": \"power_linear\",\n"
         << "    \"incoherent_power_semantics\": \"sum of five independently global-SVD-filtered per-angle linear powers; temporal bases may differ by angle\",\n"
         << "    \"angle_coherence_semantics\": \"pre_svd sum_frame_abs2(sum_angle_iq) divided by angle_count times sum_frame_angle_abs2(iq), bounded to [0,1]\",\n"
         << "    \"global_temporal_svd\": true,\n"
         << "    \"svd_keep_start\": " << config.svd.keep_start << ",\n"
         << "    \"svd_keep_end\": " << config.svd.keep_end << ",\n"
         << "    \"svd_kept_components\": "
         << config.svd.keep_end - config.svd.keep_start + 1 << ",\n"
         << "    \"svd_indexing\": \"one_based_inclusive\",\n"
         << "    \"motion_correction\": false,\n"
         << "    \"block_svd\": false,\n"
         << "    \"channel_repair\": false\n"
         << "  },\n"
         << "  \"artifacts\": {\n"
         << "    \"power_linear\": {\"path\": \"" << kPowerFile
         << "\", \"dtype\": \"float32_le\", \"shape\": ["
         << config.image_grid.nz << ", " << config.image_grid.nx
         << "], \"layout\": [\"z\", \"x\"], \"size_bytes\": "
         << expected_bytes_for<float>(pixels) << "},\n"
         << "    \"singular_values\": {\"path\": \""
         << kSingularValuesFile
         << "\", \"dtype\": \"float32_le\", \"shape\": ["
         << config.input.frame_count
         << "], \"layout\": [\"component\"], \"size_bytes\": "
         << expected_bytes_for<float>(config.input.frame_count) << "},\n";
    append_iq_artifact(
        json,
        "beamformed_iq",
        kBeamformedIqFile,
        spec.dump_beamformed_iq,
        config.input.frame_count,
        config.image_grid.nz,
        config.image_grid.nx);
    json << ",\n";
    append_angle_diagnostic_artifacts(
        json,
        spec.angle_diagnostics_enabled,
        spec.dump_angle_beamformed_iq,
        config.input.frame_count,
        config.input.sequences,
        config.image_grid.nz,
        config.image_grid.nx);
    json << ",\n";
    append_iq_artifact(
        json,
        "filtered_iq",
        kFilteredIqFile,
        spec.dump_filtered_iq,
        config.input.frame_count,
        config.image_grid.nz,
        config.image_grid.nx);
    json << "\n  },\n"
         << "  \"software\": {\"git_commit\": \""
         << json_escape(CAROTID_GIT_COMMIT)
         << "\", \"project_version\": \""
         << json_escape(CAROTID_PROJECT_VERSION)
         << "\", \"build_type\": \"" << json_escape(CAROTID_BUILD_TYPE)
         << "\", \"cuda_runtime\": \""
         << json_escape(artifacts.software.cuda_runtime)
         << "\", \"cuda_driver\": \""
         << json_escape(artifacts.software.cuda_driver)
         << "\", \"gpu_name\": \""
         << json_escape(artifacts.software.gpu_name) << "\"},\n"
         << "  \"timings_ms\": {\n"
         << "    \"read_and_map\": " << timings.read_and_map_ms << ",\n"
         << "    \"host_to_device\": " << timings.host_to_device_ms << ",\n"
         << "    \"tgc\": " << timings.tgc_ms << ",\n"
         << "    \"bandpass\": " << timings.bandpass_ms << ",\n"
         << "    \"rf_to_iq\": " << timings.rf_to_iq_ms << ",\n"
         << "    \"beamforming\": " << timings.beamforming_ms << ",\n"
         << "    \"beamformed_to_host\": "
         << timings.beamformed_to_host_ms << ",\n"
         << "    \"angle_svd_total\": "
         << timings.angle_svd_total_ms << ",\n"
         << "    \"covariance\": "
         << timings.global_svd.covariance_ms << ",\n"
         << "    \"svd\": " << timings.global_svd.eigensolve_ms << ",\n"
         << "    \"projection\": "
         << timings.global_svd.projection_ms << ",\n"
         << "    \"power\": " << timings.global_svd.power_ms << ",\n"
         << "    \"filtered_reconstruction\": "
         << timings.global_svd.filtered_reconstruction_ms << ",\n"
         << "    \"svd_device_total\": "
         << timings.global_svd.total_ms << ",\n"
         << "    \"svd_host_overhead\": "
         << timings.svd_host_overhead_ms << ",\n"
         << "    \"write\": " << timings.write_ms << ",\n"
         << "    \"total\": " << timings.total_ms << "\n"
         << "  }\n"
         << "}\n";
    return json.str();
}

void remove_incomplete_contract_outputs(
    const std::filesystem::path& run_directory) noexcept {
    const std::vector<std::string_view> file_names{
        kPowerFile,
        kSingularValuesFile,
        kManifestFile,
    };
    std::error_code error;
    for (const auto file_name : file_names) {
        const std::filesystem::path target = run_directory / file_name;
        std::filesystem::remove(target, error);
        error.clear();
        std::filesystem::path partial = target;
        partial += ".partial";
        std::filesystem::remove(partial, error);
        error.clear();
    }
    std::filesystem::remove(run_directory, error);
}

void remove_incomplete_real_outputs(
    const std::filesystem::path& run_directory) noexcept {
    const std::vector<std::string_view> file_names{
        kPowerFile,
        kSingularValuesFile,
        kBeamformedIqFile,
        kAngleBeamformedIqFile,
        kAnglePowerFile,
        kAngleSingularValuesFile,
        kIncoherentPowerFile,
        kAngleCoherenceFile,
        kFilteredIqFile,
        kManifestFile,
    };
    std::error_code error;
    for (const auto file_name : file_names) {
        const std::filesystem::path target = run_directory / file_name;
        std::filesystem::remove(target, error);
        error.clear();
        std::filesystem::path partial = target;
        partial += ".partial";
        std::filesystem::remove(partial, error);
        error.clear();
    }
    std::filesystem::remove(run_directory, error);
}

}  // namespace

BandpassFileIdentity capture_bandpass_identity(
    const BandpassConfig& bandpass) {
    BandpassFileIdentity identity;
    identity.enabled = bandpass.enabled;
    identity.coefficients_path = normalized_bandpass_path(bandpass);
    identity.dtype = bandpass.dtype;
    identity.taps = bandpass.taps;
    if (!bandpass.enabled) {
        return identity;
    }

    std::error_code error;
    identity.size_bytes =
        std::filesystem::file_size(identity.coefficients_path, error);
    if (error) {
        throw std::runtime_error(
            "failed to fingerprint FIR coefficient file: " +
            error.message());
    }
    identity.last_write_time_utc =
        file_last_write_time_utc(identity.coefficients_path);
    identity.fingerprint_present = true;
    return identity;
}

void write_complete_synthetic_run(
    const SyntheticRunSpec& spec,
    const SyntheticArtifacts& artifacts) {
    validate_artifacts(spec, artifacts);
    prepare_run_directory(spec.run_directory);
    try {
        write_binary_atomic(spec.run_directory / kPowerFile, artifacts.power);
        write_binary_atomic(
            spec.run_directory / kSingularValuesFile, artifacts.singular_values);
        const std::string manifest = build_manifest(spec, artifacts);
        write_text_atomic(spec.run_directory / kManifestFile, manifest);
    } catch (...) {
        remove_incomplete_contract_outputs(spec.run_directory);
        throw;
    }
}

void write_complete_real_run(
    const RealRunSpec& spec,
    const RealRunArtifacts& artifacts) {
    validate_real_artifacts(spec, artifacts);
    prepare_run_directory(spec.run_directory);
    try {
        const auto write_begin = std::chrono::steady_clock::now();
        write_binary_atomic(
            spec.run_directory / kPowerFile,
            artifacts.power_linear);
        write_binary_atomic(
            spec.run_directory / kSingularValuesFile,
            artifacts.singular_values);
        if (spec.dump_beamformed_iq) {
            write_binary_atomic(
                spec.run_directory / kBeamformedIqFile,
                artifacts.beamformed_iq);
        }
        if (spec.dump_angle_beamformed_iq) {
            write_binary_atomic(
                spec.run_directory / kAngleBeamformedIqFile,
                artifacts.angle_beamformed_iq);
        }
        if (spec.angle_diagnostics_enabled) {
            write_binary_atomic(
                spec.run_directory / kAnglePowerFile,
                artifacts.angle_power_linear);
            write_binary_atomic(
                spec.run_directory / kAngleSingularValuesFile,
                artifacts.angle_singular_values);
            write_binary_atomic(
                spec.run_directory / kIncoherentPowerFile,
                artifacts.incoherent_power_linear);
            write_binary_atomic(
                spec.run_directory / kAngleCoherenceFile,
                artifacts.angle_coherence);
        }
        if (spec.dump_filtered_iq) {
            write_binary_atomic(
                spec.run_directory / kFilteredIqFile,
                artifacts.filtered_iq);
        }
        const auto write_end = std::chrono::steady_clock::now();
        PipelineTimings timings = artifacts.timings;
        timings.write_ms = std::chrono::duration<double, std::milli>(
            write_end - write_begin).count();
        timings.total_ms += timings.write_ms;
        const std::string manifest =
            build_real_manifest(spec, artifacts, timings);
        write_text_atomic(spec.run_directory / kManifestFile, manifest);
    } catch (...) {
        remove_incomplete_real_outputs(spec.run_directory);
        throw;
    }
}

void write_failed_real_run_log(
    const std::filesystem::path& run_directory,
    const std::string& error_message) {
    // The caller has already established that this path did not exist before
    // the failed attempt, so removing only the known contract artifacts cannot
    // overwrite an earlier immutable run.
    if (std::filesystem::exists(run_directory) &&
        !std::filesystem::is_directory(run_directory)) {
        throw std::runtime_error(
            "failed-run path is not a directory: " + run_directory.string());
    }
    const std::filesystem::path target = run_directory / "run_error.txt";
    if (std::filesystem::exists(run_directory / kManifestFile) ||
        std::filesystem::exists(target)) {
        throw std::runtime_error(
            "refusing to replace an existing completed or failed run: " +
            run_directory.string());
    }
    remove_incomplete_real_outputs(run_directory);
    if (!std::filesystem::exists(run_directory) &&
        !std::filesystem::create_directories(run_directory)) {
        throw std::runtime_error(
            "failed to create failed-run directory: " +
            run_directory.string());
    }

    const std::string contents =
        "status=failed\nerror_message:\n" + error_message + "\n";
    try {
        write_text_atomic(target, contents);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::path partial = target;
        partial += ".partial";
        std::filesystem::remove(partial, ignored);
        throw;
    }
}

}  // namespace carotid_doppler
