#pragma once

#include <cuda_runtime_api.h>
#include <cufft.h>

#include <stdexcept>
#include <string>
#include <string_view>

namespace carotid_doppler {

inline void check_cuda(cudaError_t status, std::string_view context) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(context) + ": " + cudaGetErrorString(status));
    }
}

inline void check_cufft(cufftResult status, std::string_view context) {
    if (status != CUFFT_SUCCESS) {
        throw std::runtime_error(
            std::string(context) + ": cuFFT error " +
            std::to_string(static_cast<int>(status)));
    }
}

}  // namespace carotid_doppler
