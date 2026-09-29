#pragma once

#include "carotid_doppler/config.hpp"
#include "carotid_doppler/global_temporal_svd.hpp"
#include "carotid_doppler/rf_reader.hpp"
#include "carotid_doppler/synthetic_generator.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace carotid_doppler {

struct SyntheticRunSpec {
    std::filesystem::path run_directory;
    std::string run_id;
    std::size_t nx;
    std::size_t nz;
    std::size_t frames;
};

void write_complete_synthetic_run(
    const SyntheticRunSpec& spec,
    const SyntheticArtifacts& artifacts);

struct RuntimeSoftware {
    std::string cuda_runtime;
    std::string cuda_driver;
    std::string gpu_name;
};

struct PipelineTimings {
    double read_and_map_ms{};
    double host_to_device_ms{};
    double tgc_ms{};
    double bandpass_ms{};
    double rf_to_iq_ms{};
    double beamforming_ms{};
    double beamformed_to_host_ms{};
    double angle_svd_total_ms{};
    double svd_host_overhead_ms{};
    GlobalTemporalSvdTimings global_svd;
    double write_ms{};
    double total_ms{};
};

struct BandpassFileIdentity {
    bool enabled{};
    std::filesystem::path coefficients_path;
    std::string dtype;
    std::size_t taps{};
    bool fingerprint_present{};
    std::uintmax_t size_bytes{};
    std::string last_write_time_utc;
};

// Captures the FIR identity immediately after coefficients are loaded. When
// disabled, this normalizes only the declared path and never opens the file.
BandpassFileIdentity capture_bandpass_identity(
    const BandpassConfig& bandpass);

struct RealRunSpec {
    std::filesystem::path run_directory;
    std::string run_id;
    PipelineConfig config;
    RfFileFacts input_facts;
    BandpassFileIdentity bandpass_identity;
    std::size_t batch_frames{};
    bool dump_beamformed_iq{};
    bool angle_diagnostics_enabled{};
    bool dump_angle_beamformed_iq{};
    bool dump_filtered_iq{};
};

struct RealRunArtifacts {
    std::vector<float> power_linear;
    std::vector<float> singular_values;
    std::vector<Complex32> beamformed_iq;
    std::vector<Complex32> angle_beamformed_iq;
    std::vector<float> angle_power_linear;
    std::vector<float> angle_singular_values;
    std::vector<float> incoherent_power_linear;
    std::vector<float> angle_coherence;
    std::vector<Complex32> filtered_iq;
    RuntimeSoftware software;
    PipelineTimings timings;
};

// Writes all numeric artifacts through *.partial files, verifies their byte
// counts, and publishes run_manifest.json last. The destination directory must
// not already exist and is removed if publication fails.
void write_complete_real_run(
    const RealRunSpec& spec,
    const RealRunArtifacts& artifacts);

// Call only when the destination did not exist before the failed formal run.
// Removes this attempt's known contract artifacts, then leaves exactly one
// explicit failure record instead of a complete manifest.
void write_failed_real_run_log(
    const std::filesystem::path& run_directory,
    const std::string& error_message);

}  // namespace carotid_doppler
