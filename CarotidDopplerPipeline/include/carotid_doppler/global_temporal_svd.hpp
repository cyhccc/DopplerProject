#pragma once

#include <cstddef>
#include <vector>

namespace carotid_doppler {

// Binary-compatible with one interleaved complex float32 sample on disk and GPU.
struct alignas(8) Complex32 {
    float real;
    float imag;
};

struct GlobalTemporalSvdConfig {
    std::size_t pixels;
    std::size_t frames;
    std::size_t keep_start;
    std::size_t keep_end;
};

struct GlobalTemporalSvdTimings {
    double covariance_ms{};
    double eigensolve_ms{};
    double projection_ms{};
    double power_ms{};
    double filtered_reconstruction_ms{};
    double total_ms{};
};

struct GlobalTemporalSvdResult {
    std::vector<float> singular_values;
    std::vector<Complex32> filtered_iq;
    std::vector<float> power_linear;
    GlobalTemporalSvdTimings timings;
};

// Input and optional filtered output use [frame, pixel] storage:
// index = pixel + frame * pixels. This is the column-major representation of
// the Casorati matrix A[pixel, frame]. keep_start/keep_end are 1-based inclusive.
// Device pointers must refer to allocations large enough for the declared sizes.
// d_filtered_iq may be null when the diagnostic IQ is not required.
void global_temporal_svd_device(
    const Complex32* d_input_iq,
    Complex32* d_filtered_iq,
    float* d_singular_values,
    float* d_power_linear,
    const GlobalTemporalSvdConfig& config,
    GlobalTemporalSvdTimings* timings = nullptr);

// Host convenience seam used by deterministic diagnostics and NumPy comparison.
// The filtered IQ is copied back only when include_filtered_iq is true.
GlobalTemporalSvdResult global_temporal_svd_host(
    const Complex32* input_iq,
    std::size_t input_count,
    const GlobalTemporalSvdConfig& config,
    bool include_filtered_iq);

}  // namespace carotid_doppler
