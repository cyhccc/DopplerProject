#pragma once

#include "carotid_doppler/rf_bandpass.hpp"

#include <cuda_runtime_api.h>
#include <vector_types.h>

#include <cstddef>
#include <memory>

namespace carotid_doppler {

// Builds the analytic signal IQ = RF + j*Hilbert(RF) independently along the
// contiguous sample axis of every [frame, sequence, channel] trace.
class RfToIqPlan {
public:
    explicit RfToIqPlan(RfTensorShape shape);
    ~RfToIqPlan();

    RfToIqPlan(const RfToIqPlan&) = delete;
    RfToIqPlan& operator=(const RfToIqPlan&) = delete;
    RfToIqPlan(RfToIqPlan&&) noexcept;
    RfToIqPlan& operator=(RfToIqPlan&&) noexcept;

    void transform(
        const float* device_rf,
        float2* device_iq,
        cudaStream_t stream = nullptr);

    const RfTensorShape& shape() const noexcept;
    std::size_t workspace_bytes() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace carotid_doppler
