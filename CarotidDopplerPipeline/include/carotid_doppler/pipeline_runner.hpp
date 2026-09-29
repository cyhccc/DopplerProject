#pragma once

#include "carotid_doppler/config.hpp"

#include <cstddef>
#include <filesystem>

namespace carotid_doppler {

struct RealPipelineOptions {
    std::filesystem::path run_directory;
    std::size_t batch_frames{4};
    bool dump_beamformed_iq{false};
    bool angle_diagnostics_enabled{false};
    bool dump_angle_beamformed_iq{false};
    bool dump_filtered_iq{false};
};

// Enforces the fixed 381-frame five-angle carotid production protocol. The
// JSON parser and low-level diagnostics intentionally remain dimension-generic.
void validate_real_carotid_protocol(const PipelineConfig& config);

// Executes the numeric-only production pipeline. RF is read and processed in
// small frame batches; only the canonical [frame,nz,nx] beamformed IQ cache is
// retained across the complete frame window for the one global temporal SVD.
// The function never performs dB conversion or image rendering.
void run_real_pipeline(
    const PipelineConfig& config,
    const RealPipelineOptions& options);

}  // namespace carotid_doppler
