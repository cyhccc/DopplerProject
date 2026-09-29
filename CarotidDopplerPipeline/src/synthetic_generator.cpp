#include "carotid_doppler/synthetic_generator.hpp"

#include <cuda_runtime_api.h>

#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

extern "C" cudaError_t carotid_generate_synthetic_cuda(
    float* host_power,
    int nx,
    int pixel_count,
    float* host_singular_values,
    int frames);

namespace carotid_doppler {
namespace {

void check_cuda(cudaError_t result, const char* operation) {
    if (result == cudaSuccess) {
        return;
    }

    std::ostringstream message;
    message << operation << " failed: " << cudaGetErrorString(result);
    throw std::runtime_error(message.str());
}

int checked_int(std::size_t value, const char* name) {
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(std::string(name) + " exceeds CUDA int range");
    }
    return static_cast<int>(value);
}

std::string runtime_version_string() {
    int version = 0;
    check_cuda(cudaRuntimeGetVersion(&version), "cudaRuntimeGetVersion");
    const int major = version / 1000;
    const int minor = (version % 1000) / 10;
    return std::to_string(major) + "." + std::to_string(minor);
}

std::string active_gpu_name() {
    int device = 0;
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
    return properties.name;
}

}  // namespace

SyntheticArtifacts generate_synthetic_artifacts(
    std::size_t nx,
    std::size_t nz,
    std::size_t frames) {
    if (nx < 2 || nz < 2 || frames == 0) {
        throw std::invalid_argument("synthetic dimensions require nx>=2, nz>=2, frames>=1");
    }
    if (nx > std::numeric_limits<std::size_t>::max() / nz) {
        throw std::invalid_argument("synthetic nx*nz overflows size_t");
    }

    const std::size_t pixel_count = nx * nz;
    SyntheticArtifacts artifacts;
    artifacts.power.resize(pixel_count);
    artifacts.singular_values.resize(frames);
    check_cuda(
        carotid_generate_synthetic_cuda(
            artifacts.power.data(),
            checked_int(nx, "nx"),
            checked_int(pixel_count, "nx*nz"),
            artifacts.singular_values.data(),
            checked_int(frames, "frames")),
        "generate deterministic synthetic artifacts");
    artifacts.cuda_runtime = runtime_version_string();
    artifacts.gpu_name = active_gpu_name();
    return artifacts;
}

}  // namespace carotid_doppler
