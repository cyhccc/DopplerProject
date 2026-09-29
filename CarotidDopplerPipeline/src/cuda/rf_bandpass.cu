#include "carotid_doppler/rf_bandpass.hpp"

#include "carotid_doppler/cuda_error.hpp"

#include <cuda_runtime.h>

#include <climits>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace carotid_doppler {
namespace {

__global__ void map_rf_and_tgc_kernel(
    const std::int16_t* __restrict__ input,
    float* __restrict__ output,
    std::size_t value_count,
    std::size_t samples,
    float scale_per_sample,
    float offset,
    bool enabled) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= value_count) {
        return;
    }
    float value = static_cast<float>(input[index]);
    if (enabled) {
        const std::size_t sample = index % samples;
        const float gain = expf(fmaf(
            static_cast<float>(sample + 1), scale_per_sample, offset));
        value *= gain;
    }
    output[index] = value;
}

__global__ void centered_fir_or_copy_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ coefficients,
    std::size_t value_count,
    std::size_t samples,
    int taps,
    bool enabled) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= value_count) {
        return;
    }
    if (!enabled) {
        output[index] = input[index];
        return;
    }

    const std::size_t trace_base = (index / samples) * samples;
    const int sample = static_cast<int>(index - trace_base);
    const int center = taps / 2;
    float sum = 0.0F;
    for (int tap = 0; tap < taps; ++tap) {
        const int source_sample = sample + tap - center;
        if (source_sample >= 0 && source_sample < static_cast<int>(samples)) {
            sum = fmaf(
                coefficients[tap],
                input[trace_base + static_cast<std::size_t>(source_sample)],
                sum);
        }
    }
    output[index] = sum;
}

}  // namespace

void map_rf_and_apply_tgc(
    const std::int16_t* device_input,
    float* device_output,
    const RfTensorShape& shape,
    const RfTgcParameters& parameters,
    cudaStream_t stream) {
    shape.validate();
    if (device_input == nullptr || device_output == nullptr) {
        throw RfProcessingError("RF TGC device input/output must not be null");
    }
    if (!std::isfinite(parameters.scale_per_sample) ||
        !std::isfinite(parameters.offset) ||
        parameters.scale_per_sample < 0.0F) {
        throw RfProcessingError("RF TGC parameters must be finite");
    }
    constexpr unsigned int threads = 256;
    const std::size_t values = shape.value_count();
    const std::size_t block_count = (values + threads - 1) / threads;
    if (block_count > static_cast<std::size_t>(UINT_MAX)) {
        throw RfProcessingError("RF TGC launch grid exceeds CUDA x dimension");
    }
    map_rf_and_tgc_kernel<<<
        static_cast<unsigned int>(block_count), threads, 0, stream>>>(
        device_input,
        device_output,
        values,
        shape.samples,
        parameters.scale_per_sample,
        parameters.offset,
        parameters.enabled);
    check_cuda(cudaGetLastError(), "launch RF map and TGC");
}

void apply_centered_rf_bandpass(
    const float* device_input,
    float* device_output,
    const RfTensorShape& shape,
    const float* device_coefficients,
    std::size_t taps,
    bool enabled,
    cudaStream_t stream) {
    shape.validate();
    if (device_input == nullptr || device_output == nullptr) {
        throw RfProcessingError("RF bandpass device input/output must not be null");
    }
    if (device_input == device_output) {
        throw RfProcessingError("RF bandpass input/output must not alias");
    }
    if (shape.samples > static_cast<std::size_t>(INT_MAX)) {
        throw RfProcessingError("RF sample count exceeds CUDA kernel int range");
    }
    if (enabled) {
        if (device_coefficients == nullptr) {
            throw RfProcessingError(
                "enabled RF bandpass requires device FIR coefficients");
        }
        if (taps != kRequiredBandpassTaps) {
            throw RfProcessingError("enabled RF bandpass requires exactly 129 taps");
        }
    }

    constexpr unsigned int threads = 256;
    const std::size_t values = shape.value_count();
    const std::size_t block_count = (values + threads - 1) / threads;
    if (block_count > static_cast<std::size_t>(UINT_MAX)) {
        throw RfProcessingError("RF bandpass launch grid exceeds CUDA x dimension");
    }
    centered_fir_or_copy_kernel<<<
        static_cast<unsigned int>(block_count), threads, 0, stream>>>(
        device_input,
        device_output,
        device_coefficients,
        values,
        shape.samples,
        static_cast<int>(taps),
        enabled);
    check_cuda(cudaGetLastError(), "launch centered RF bandpass");
}

}  // namespace carotid_doppler
