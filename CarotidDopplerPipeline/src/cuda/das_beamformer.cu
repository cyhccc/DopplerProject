#include "carotid_doppler/das_beamformer.hpp"

#include "carotid_doppler/cuda_error.hpp"

#include <cuda_runtime.h>

#include <climits>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <utility>

namespace carotid_doppler {
namespace {

constexpr unsigned int kFrameThreads = 32;
constexpr double kPi = 3.141592653589793238462643383279502884;

std::size_t checked_multiply(
    std::size_t left,
    std::size_t right,
    const char* label) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        throw DasBeamformingError(std::string(label) + " overflows size_t");
    }
    return left * right;
}

float checked_float(double value, const char* label, bool positive) {
    if (!std::isfinite(value) || (positive && !(value > 0.0)) ||
        value < -static_cast<double>(std::numeric_limits<float>::max()) ||
        value > static_cast<double>(std::numeric_limits<float>::max())) {
        throw DasBeamformingError(
            std::string(label) + " cannot be represented as finite float32");
    }
    return static_cast<float>(value);
}

struct KernelParameters {
    std::size_t frames;
    std::size_t sequences;
    std::size_t channels;
    std::size_t samples;
    std::size_t nx;
    std::size_t nz;
    float samples_per_metre;
    float pitch_m;
    float tx_delay_offset_samples;
    float x_min_m;
    float x_step_m;
    float z_min_m;
    float z_step_m;
    float f_number;
    float minimum_aperture_m;
    int aperture_edge_reference;
    int tukey25_apodization;
};

__device__ __forceinline__ float tukey25_aperture_weight(
    float normalized_distance) {
    constexpr float alpha = 0.25F;
    constexpr float flat_limit = 1.0F - alpha;
    if (normalized_distance > 1.0F) {
        return 0.0F;
    }
    if (normalized_distance <= flat_limit) {
        return 1.0F;
    }
    const float taper_position =
        (normalized_distance - flat_limit) / alpha;
    return 0.5F *
        (1.0F + cosf(3.14159265358979323846F * taper_position));
}

__device__ __forceinline__ float2 sample_linear(
    const float2* input,
    std::size_t trace_base,
    float sample_position) {
    const std::size_t sample0 = static_cast<std::size_t>(sample_position);
    const float fraction = sample_position - static_cast<float>(sample0);
    const float2 value0 = input[trace_base + sample0];
    const float2 value1 = input[trace_base + sample0 + 1];
    return make_float2(
        fmaf(fraction, value1.x - value0.x, value0.x),
        fmaf(fraction, value1.y - value0.y, value0.y));
}

__global__ void das_beamform_kernel(
    const float2* __restrict__ input_iq,
    float2* __restrict__ output_iq,
    float2* __restrict__ per_angle_output_iq,
    const float* __restrict__ transmit_angles_rad,
    KernelParameters parameters) {
    const std::size_t pixel = static_cast<std::size_t>(blockIdx.x);
    const std::size_t pixels = parameters.nx * parameters.nz;
    if (pixel >= pixels) {
        return;
    }

    const std::size_t x_index = pixel % parameters.nx;
    const std::size_t z_index = pixel / parameters.nx;
    const float pixel_x =
        parameters.x_min_m + static_cast<float>(x_index) * parameters.x_step_m;
    const float pixel_z =
        parameters.z_min_m + static_cast<float>(z_index) * parameters.z_step_m;
    const float element_x_origin =
        -0.5F * static_cast<float>(parameters.channels - 1) *
        parameters.pitch_m;

    extern __shared__ float shared_delays[];
    float* const receive_delay_samples = shared_delays;
    float* const receive_weights =
        shared_delays + parameters.channels;
    float* const transmit_delay_samples =
        shared_delays + 2 * parameters.channels;

    const float aperture_diameter = fmaxf(
        pixel_z / parameters.f_number, parameters.minimum_aperture_m);
    const float aperture_radius = 0.5F * aperture_diameter;
    for (std::size_t channel = threadIdx.x; channel < parameters.channels;
         channel += blockDim.x) {
        const float element_x =
            element_x_origin + static_cast<float>(channel) * parameters.pitch_m;
        const float lateral_distance = pixel_x - element_x;
        const float normalized_distance =
            fabsf(lateral_distance) / aperture_radius;
        const float weight = parameters.tukey25_apodization != 0
            ? tukey25_aperture_weight(normalized_distance)
            : (normalized_distance <= 1.0F ? 1.0F : 0.0F);
        receive_weights[channel] = weight;
        receive_delay_samples[channel] = weight > 0.0F
            ? hypotf(lateral_distance, pixel_z) * parameters.samples_per_metre
            : -1.0F;
    }

    for (std::size_t sequence = threadIdx.x;
         sequence < parameters.sequences;
         sequence += blockDim.x) {
        const float angle = transmit_angles_rad[sequence];
        float direction_x = 0.0F;
        float direction_z = 0.0F;
        sincosf(angle, &direction_x, &direction_z);

        float reference_distance = 0.0F;
        if (parameters.aperture_edge_reference != 0) {
            const float half_aperture =
                0.5F * static_cast<float>(parameters.channels - 1) *
                parameters.pitch_m;
            const float reference_edge_x =
                direction_x >= 0.0F ? -half_aperture : half_aperture;
            reference_distance = reference_edge_x * direction_x;
        }
        transmit_delay_samples[sequence] =
            (pixel_z * direction_z + pixel_x * direction_x -
             reference_distance) *
                parameters.samples_per_metre -
            parameters.tx_delay_offset_samples;
    }
    __syncthreads();

    for (std::size_t frame = threadIdx.x; frame < parameters.frames;
         frame += blockDim.x) {
        float2 coherent_sum = make_float2(0.0F, 0.0F);
        for (std::size_t sequence = 0; sequence < parameters.sequences;
             ++sequence) {
            float2 per_angle_sum = make_float2(0.0F, 0.0F);
            for (std::size_t channel = 0; channel < parameters.channels;
                 ++channel) {
                const float receive_delay = receive_delay_samples[channel];
                if (receive_delay < 0.0F) {
                    continue;
                }
                const float delay =
                    transmit_delay_samples[sequence] + receive_delay;
                if (delay < 0.0F ||
                    delay >= static_cast<float>(parameters.samples - 1)) {
                    continue;
                }
                const std::size_t trace_base =
                    (((frame * parameters.sequences + sequence) *
                      parameters.channels +
                      channel) *
                     parameters.samples);
                const float2 sample =
                    sample_linear(input_iq, trace_base, delay);
                const float weight = receive_weights[channel];
                coherent_sum.x = fmaf(sample.x, weight, coherent_sum.x);
                coherent_sum.y = fmaf(sample.y, weight, coherent_sum.y);
                per_angle_sum.x = fmaf(sample.x, weight, per_angle_sum.x);
                per_angle_sum.y = fmaf(sample.y, weight, per_angle_sum.y);
            }
            if (per_angle_output_iq != nullptr) {
                const std::size_t output_index =
                    (frame * parameters.sequences + sequence) * pixels + pixel;
                per_angle_output_iq[output_index] = per_angle_sum;
            }
        }
        // Contract: x + z*nx + frame*nx*nz.
        output_iq[frame * pixels + pixel] = coherent_sum;
    }
}

}  // namespace

std::size_t DasTensorShape::input_value_count() const {
    validate();
    return checked_multiply(
        checked_multiply(
            checked_multiply(frames, sequences, "DAS input value count"),
            channels,
            "DAS input value count"),
        samples,
        "DAS input value count");
}

std::size_t DasTensorShape::pixel_count() const {
    validate();
    return checked_multiply(nx, nz, "DAS pixel count");
}

std::size_t DasTensorShape::output_value_count() const {
    return checked_multiply(frames, pixel_count(), "DAS output value count");
}

std::size_t DasTensorShape::per_angle_output_value_count() const {
    return checked_multiply(
        sequences,
        output_value_count(),
        "DAS per-angle output value count");
}

void DasTensorShape::validate() const {
    if (frames == 0 || sequences == 0 || channels == 0 || samples < 2 ||
        nx < 2 || nz < 2) {
        throw DasBeamformingError(
            "DAS dimensions require frames/sequences/channels > 0, "
            "samples >= 2, and nx/nz >= 2");
    }
}

void DasParameters::validate() const {
    shape.validate();
    if (transmit_angles_rad.size() != shape.sequences) {
        throw DasBeamformingError(
            "DAS transmit angle count must equal the sequence count");
    }
    for (std::size_t index = 0; index < transmit_angles_rad.size(); ++index) {
        if (!std::isfinite(transmit_angles_rad[index])) {
            throw DasBeamformingError(
                "DAS transmit angle is not finite at sequence " +
                std::to_string(index));
        }
    }
    if (!(sample_rate_hz > 0.0F) || !(sound_speed_m_per_s > 0.0F) ||
        !(pitch_m > 0.0F) || !(f_number > 0.0F) ||
        !(minimum_aperture_m > 0.0F) ||
        !std::isfinite(tx_delay_offset_samples) ||
        tx_delay_offset_samples < 0.0F || !std::isfinite(x_min_m) ||
        !std::isfinite(x_max_m) || !std::isfinite(z_min_m) ||
        !std::isfinite(z_max_m) || !(x_min_m < x_max_m) ||
        !(z_min_m < z_max_m)) {
        throw DasBeamformingError("DAS geometry contains an invalid value");
    }
}

DasParameters make_das_parameters(
    const PipelineConfig& config,
    std::size_t frame_count) {
    DasParameters parameters;
    parameters.shape = DasTensorShape{
        frame_count,
        config.input.sequences,
        config.input.channels,
        config.input.samples,
        config.image_grid.nx,
        config.image_grid.nz};
    parameters.sample_rate_hz = checked_float(
        config.acquisition.sample_rate_hz, "sample_rate_hz", true);
    parameters.sound_speed_m_per_s = checked_float(
        config.acquisition.sound_speed_m_per_s,
        "sound_speed_m_per_s",
        true);
    parameters.pitch_m =
        checked_float(config.acquisition.pitch_m, "pitch_m", true);
    parameters.transmit_angles_rad.reserve(
        config.acquisition.transmit_angles_deg.size());
    for (const double angle_deg : config.acquisition.transmit_angles_deg) {
        parameters.transmit_angles_rad.push_back(checked_float(
            angle_deg * kPi / 180.0, "transmit angle", false));
    }
    parameters.tx_delay_offset_samples = checked_float(
        static_cast<double>(config.acquisition.tx_delay_offset_samples),
        "tx_delay_offset_samples",
        false);
    if (config.acquisition.tx_reference_mode == "aperture_edge") {
        parameters.tx_reference_mode = TxReferenceMode::aperture_edge;
    } else if (config.acquisition.tx_reference_mode == "aligned_reference") {
        parameters.tx_reference_mode = TxReferenceMode::aligned_reference;
    } else {
        throw DasBeamformingError(
            "unsupported tx_reference_mode: " +
            config.acquisition.tx_reference_mode);
    }
    parameters.x_min_m =
        checked_float(config.image_grid.x_min_m, "x_min_m", false);
    parameters.x_max_m =
        checked_float(config.image_grid.x_max_m, "x_max_m", false);
    parameters.z_min_m =
        checked_float(config.image_grid.z_min_m, "z_min_m", false);
    parameters.z_max_m =
        checked_float(config.image_grid.z_max_m, "z_max_m", false);
    parameters.f_number =
        checked_float(config.receive.f_number, "f_number", true);
    parameters.minimum_aperture_m = checked_float(
        config.receive.minimum_aperture_m, "minimum_aperture_m", true);
    if (config.receive.apodization == "rectangular") {
        parameters.receive_apodization = ReceiveApodization::rectangular;
    } else if (config.receive.apodization == "tukey25") {
        parameters.receive_apodization = ReceiveApodization::tukey25;
    } else {
        throw DasBeamformingError(
            "unsupported receive apodization: " +
            config.receive.apodization);
    }
    parameters.validate();
    return parameters;
}

struct DasBeamformer::Impl {
    explicit Impl(DasParameters requested_parameters)
        : parameters(std::move(requested_parameters)) {
        parameters.validate();
        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&device_angles),
                parameters.transmit_angles_rad.size() * sizeof(float)),
            "allocate DAS transmit angles");
        try {
            check_cuda(
                cudaMemcpy(
                    device_angles,
                    parameters.transmit_angles_rad.data(),
                    parameters.transmit_angles_rad.size() * sizeof(float),
                    cudaMemcpyHostToDevice),
                "copy DAS transmit angles to device");
        } catch (...) {
            cudaFree(device_angles);
            device_angles = nullptr;
            throw;
        }
    }

    ~Impl() {
        if (device_angles != nullptr) {
            cudaFree(device_angles);
        }
    }

    DasParameters parameters;
    float* device_angles{};
};

DasBeamformer::DasBeamformer(DasParameters parameters)
    : impl_(std::make_unique<Impl>(std::move(parameters))) {}

DasBeamformer::~DasBeamformer() = default;
DasBeamformer::DasBeamformer(DasBeamformer&&) noexcept = default;
DasBeamformer& DasBeamformer::operator=(DasBeamformer&&) noexcept = default;

void DasBeamformer::beamform(
    const float2* device_input_iq,
    float2* device_output_iq,
    cudaStream_t stream) const {
    beamform_with_per_angle(
        device_input_iq, device_output_iq, nullptr, stream);
}

void DasBeamformer::beamform_with_per_angle(
    const float2* device_input_iq,
    float2* device_output_iq,
    float2* device_per_angle_output_iq,
    cudaStream_t stream) const {
    if (device_input_iq == nullptr || device_output_iq == nullptr) {
        throw DasBeamformingError("DAS device input/output must not be null");
    }
    const DasParameters& p = impl_->parameters;
    p.validate();
    const std::size_t pixels = p.shape.pixel_count();
    if (pixels > static_cast<std::size_t>(INT_MAX)) {
        throw DasBeamformingError("DAS pixel count exceeds CUDA grid.x range");
    }
    const std::size_t shared_values =
        2 * p.shape.channels + p.shape.sequences;
    if (shared_values >
        std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        throw DasBeamformingError("DAS shared delay workspace overflows size_t");
    }

    const KernelParameters kernel_parameters{
        p.shape.frames,
        p.shape.sequences,
        p.shape.channels,
        p.shape.samples,
        p.shape.nx,
        p.shape.nz,
        p.sample_rate_hz / p.sound_speed_m_per_s,
        p.pitch_m,
        p.tx_delay_offset_samples,
        p.x_min_m,
        (p.x_max_m - p.x_min_m) / static_cast<float>(p.shape.nx - 1),
        p.z_min_m,
        (p.z_max_m - p.z_min_m) / static_cast<float>(p.shape.nz - 1),
        p.f_number,
        p.minimum_aperture_m,
        p.tx_reference_mode == TxReferenceMode::aperture_edge ? 1 : 0,
        p.receive_apodization == ReceiveApodization::tukey25 ? 1 : 0};
    das_beamform_kernel<<<
        static_cast<unsigned int>(pixels),
        kFrameThreads,
        shared_values * sizeof(float),
        stream>>>(
        device_input_iq,
        device_output_iq,
        device_per_angle_output_iq,
        impl_->device_angles,
        kernel_parameters);
    check_cuda(cudaGetLastError(), "launch DAS beamformer");
}

const DasParameters& DasBeamformer::parameters() const noexcept {
    return impl_->parameters;
}

}  // namespace carotid_doppler
