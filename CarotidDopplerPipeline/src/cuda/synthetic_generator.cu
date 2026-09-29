#include "carotid_doppler/cuda_error.hpp"
#include "carotid_doppler/device_buffer.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <string>
#include <string_view>

namespace {

__global__ void generate_power_kernel(float* power, int nx, int pixel_count) {
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixel_count) {
        return;
    }

    const int z = pixel / nx;
    const int x = pixel - z * nx;
    power[pixel] = static_cast<float>((z + 1) * (x + 1));
}

__global__ void generate_singular_values_kernel(float* values, int frames) {
    const int component = blockIdx.x * blockDim.x + threadIdx.x;
    if (component < frames) {
        values[component] = static_cast<float>(frames - component);
    }
}

std::string stage_context(
    std::string_view stage,
    std::size_t power_bytes,
    std::size_t singular_values_bytes) {
    return std::string(stage) + " (power_bytes=" +
        std::to_string(power_bytes) + ", singular_values_bytes=" +
        std::to_string(singular_values_bytes) + ")";
}

}  // namespace

extern "C" cudaError_t carotid_generate_synthetic_cuda(
    float* host_power,
    int nx,
    int pixel_count,
    float* host_singular_values,
    int frames) {
    const std::size_t power_bytes =
        static_cast<std::size_t>(pixel_count) * sizeof(float);
    const std::size_t singular_values_bytes =
        static_cast<std::size_t>(frames) * sizeof(float);
    const auto context = [&](std::string_view stage) {
        return stage_context(stage, power_bytes, singular_values_bytes);
    };

    carotid_doppler::DeviceBuffer<float> device_power(
        static_cast<std::size_t>(pixel_count),
        context("synthetic power allocation"));
    carotid_doppler::DeviceBuffer<float> device_singular_values(
        static_cast<std::size_t>(frames),
        context("synthetic singular-values allocation"));

    constexpr int threads = 256;
    generate_power_kernel<<<(pixel_count + threads - 1) / threads, threads>>>(
        device_power.get(), nx, pixel_count);
    carotid_doppler::check_cuda(
        cudaGetLastError(), context("synthetic power kernel launch"));

    generate_singular_values_kernel<<<(frames + threads - 1) / threads, threads>>>(
        device_singular_values.get(), frames);
    carotid_doppler::check_cuda(
        cudaGetLastError(), context("synthetic singular-values kernel launch"));
    carotid_doppler::check_cuda(
        cudaDeviceSynchronize(), context("synthetic kernel synchronization"));

    carotid_doppler::check_cuda(
        cudaMemcpy(
            host_power,
            device_power.get(),
            power_bytes,
            cudaMemcpyDeviceToHost),
        context("synthetic power device-to-host copy"));
    carotid_doppler::check_cuda(
        cudaMemcpy(
            host_singular_values,
            device_singular_values.get(),
            singular_values_bytes,
            cudaMemcpyDeviceToHost),
        context("synthetic singular-values device-to-host copy"));

    device_singular_values.release(context("synthetic singular-values release"));
    device_power.release(context("synthetic power release"));
    return cudaSuccess;
}
