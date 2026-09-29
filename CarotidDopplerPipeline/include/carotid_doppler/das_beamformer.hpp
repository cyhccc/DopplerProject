#pragma once

#include "carotid_doppler/config.hpp"

#include <cuda_runtime_api.h>
#include <vector_types.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

namespace carotid_doppler {

class DasBeamformingError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

enum class TxReferenceMode {
    aligned_reference,
    aperture_edge,
};

enum class ReceiveApodization {
    rectangular,
    tukey25,
};

struct DasTensorShape {
    std::size_t frames{};
    std::size_t sequences{};
    std::size_t channels{};
    std::size_t samples{};
    std::size_t nx{};
    std::size_t nz{};

    std::size_t input_value_count() const;
    std::size_t pixel_count() const;
    std::size_t output_value_count() const;
    std::size_t per_angle_output_value_count() const;
    void validate() const;
};

// All SI quantities use the unit declared by the member name. Input IQ is
// always [frame, sequence, internal_channel, sample], where internal_channel
// is physical ascending_x and sample is contiguous. Output IQ is
// [frame, nz, nx], with x contiguous.
struct DasParameters {
    DasTensorShape shape;
    float sample_rate_hz{};
    float sound_speed_m_per_s{};
    float pitch_m{};
    std::vector<float> transmit_angles_rad;
    float tx_delay_offset_samples{};
    TxReferenceMode tx_reference_mode{TxReferenceMode::aligned_reference};
    float x_min_m{};
    float x_max_m{};
    float z_min_m{};
    float z_max_m{};
    float f_number{};
    float minimum_aperture_m{};
    ReceiveApodization receive_apodization{ReceiveApodization::rectangular};

    void validate() const;
};

DasParameters make_das_parameters(
    const PipelineConfig& config,
    std::size_t frame_count);

class DasBeamformer {
public:
    explicit DasBeamformer(DasParameters parameters);
    ~DasBeamformer();

    DasBeamformer(const DasBeamformer&) = delete;
    DasBeamformer& operator=(const DasBeamformer&) = delete;
    DasBeamformer(DasBeamformer&&) noexcept;
    DasBeamformer& operator=(DasBeamformer&&) noexcept;

    // No transpose or direction correction is performed here. The caller must
    // provide IQ in the canonical internal ascending_x channel order.
    void beamform(
        const float2* device_input_iq,
        float2* device_output_iq,
        cudaStream_t stream = nullptr) const;

    // Optionally exposes the receive-channel coherent sum for every transmit
    // sequence while preserving the regular all-sequence coherent output.
    // Per-angle output is [frame, sequence, nz, nx], with x contiguous.
    void beamform_with_per_angle(
        const float2* device_input_iq,
        float2* device_output_iq,
        float2* device_per_angle_output_iq,
        cudaStream_t stream = nullptr) const;

    const DasParameters& parameters() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace carotid_doppler
