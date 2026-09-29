#pragma once

#include "carotid_doppler/config.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace carotid_doppler {

inline constexpr std::size_t kRequiredBandpassTaps = 129;

class RfProcessingError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct RfTensorShape {
    std::size_t frames{};
    std::size_t sequences{};
    std::size_t channels{};
    std::size_t samples{};

    std::size_t trace_count() const;
    std::size_t value_count() const;
    void validate() const;
};

struct RfTgcParameters {
    bool enabled{};
    float scale_per_sample{};
    float offset{};
};

// Reproduces the acquisition DLL attenuation law from explicit configuration:
// alpha_db_cm = coeff * (f0_MHz ^ power), alpha_np_m = alpha_db_cm / 14.686 * 100,
// scale = alpha_np_m * c / fs, offset = -alpha_np_m * c * cycle / f0.
RfTgcParameters make_rf_tgc_parameters(const PipelineConfig& config);

// When bandpass is disabled this still validates dtype/tap semantics, but it
// deliberately does not open the coefficient file.
std::vector<float> load_bandpass_coefficients(
    const BandpassConfig& config);

// Maps int16 RF to float32 and optionally applies the DLL-matched TGC gain
// exp((sample_zero_based + 1) * scale_per_sample + offset). Disabled TGC keeps
// the exact int16 -> float32 mapping semantics.
void map_rf_and_apply_tgc(
    const std::int16_t* device_input,
    float* device_output,
    const RfTensorShape& shape,
    const RfTgcParameters& parameters,
    cudaStream_t stream = nullptr);

// Input and output both use float32
// [frame, sequence, internal_channel, sample], with sample contiguous.
// Disabled bandpass performs a float32 copy; input/output must not alias.
void apply_centered_rf_bandpass(
    const float* device_input,
    float* device_output,
    const RfTensorShape& shape,
    const float* device_coefficients,
    std::size_t taps,
    bool enabled,
    cudaStream_t stream = nullptr);

}  // namespace carotid_doppler
