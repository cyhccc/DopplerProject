#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace carotid_doppler {

struct SyntheticArtifacts {
    std::vector<float> power;
    std::vector<float> singular_values;
    std::string cuda_runtime;
    std::string gpu_name;
};

SyntheticArtifacts generate_synthetic_artifacts(
    std::size_t nx,
    std::size_t nz,
    std::size_t frames);

}  // namespace carotid_doppler
