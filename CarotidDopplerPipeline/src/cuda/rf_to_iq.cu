#include "carotid_doppler/rf_to_iq.hpp"

#include "carotid_doppler/cuda_error.hpp"

#include <cuda_runtime.h>
#include <cufft.h>

#include <climits>
#include <cstddef>
#include <limits>
#include <memory>
#include <utility>

namespace carotid_doppler {
namespace {

__global__ void hilbert_half_spectrum_kernel(
    cufftComplex* spectrum,
    int samples,
    int half_bins,
    std::size_t batch_count) {
    const std::size_t batch = blockIdx.x;
    if (batch >= batch_count) {
        return;
    }
    const std::size_t base = batch * static_cast<std::size_t>(half_bins);
    for (int bin = static_cast<int>(threadIdx.x); bin < half_bins;
         bin += static_cast<int>(blockDim.x)) {
        const bool dc_or_nyquist =
            bin == 0 || (((samples & 1) == 0) && bin == samples / 2);
        const cufftComplex input = spectrum[base + static_cast<std::size_t>(bin)];
        spectrum[base + static_cast<std::size_t>(bin)] = dc_or_nyquist
            ? make_cuFloatComplex(0.0F, 0.0F)
            : make_cuFloatComplex(input.y, -input.x);
    }
}

__global__ void pack_analytic_iq_kernel(
    const float* real,
    const float* hilbert_unnormalized,
    float2* iq,
    std::size_t value_count,
    float inverse_samples) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < value_count) {
        iq[index] = make_float2(
            real[index], hilbert_unnormalized[index] * inverse_samples);
    }
}

}  // namespace

struct RfToIqPlan::Impl {
    explicit Impl(RfTensorShape requested_shape) : shape(requested_shape) {
        shape.validate();
        const std::size_t traces = shape.trace_count();
        if (shape.samples > static_cast<std::size_t>(INT_MAX) ||
            traces > static_cast<std::size_t>(INT_MAX)) {
            throw RfProcessingError(
                "RF-to-IQ dimensions exceed cuFFT PlanMany int range");
        }
        samples = static_cast<int>(shape.samples);
        batches = static_cast<int>(traces);
        half_bins = samples / 2 + 1;

        if (traces > std::numeric_limits<std::size_t>::max() /
                         static_cast<std::size_t>(half_bins)) {
            throw RfProcessingError("RF-to-IQ half-spectrum size overflows size_t");
        }
        const std::size_t half_values =
            traces * static_cast<std::size_t>(half_bins);
        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&half_spectrum),
                half_values * sizeof(cufftComplex)),
            "allocate RF-to-IQ half spectrum");
        try {
            check_cuda(
                cudaMalloc(
                    reinterpret_cast<void**>(&hilbert),
                    shape.value_count() * sizeof(float)),
                "allocate RF-to-IQ Hilbert workspace");

            int dimensions[] = {samples};
            check_cufft(
                cufftPlanMany(
                    &forward,
                    1,
                    dimensions,
                    nullptr,
                    1,
                    samples,
                    nullptr,
                    1,
                    half_bins,
                    CUFFT_R2C,
                    batches),
                "create RF-to-IQ forward plan");
            check_cufft(
                cufftPlanMany(
                    &inverse,
                    1,
                    dimensions,
                    nullptr,
                    1,
                    half_bins,
                    nullptr,
                    1,
                    samples,
                    CUFFT_C2R,
                    batches),
                "create RF-to-IQ inverse plan");
        } catch (...) {
            release();
            throw;
        }
    }

    ~Impl() { release(); }

    void release() noexcept {
        if (inverse != 0) {
            cufftDestroy(inverse);
            inverse = 0;
        }
        if (forward != 0) {
            cufftDestroy(forward);
            forward = 0;
        }
        if (hilbert != nullptr) {
            cudaFree(hilbert);
            hilbert = nullptr;
        }
        if (half_spectrum != nullptr) {
            cudaFree(half_spectrum);
            half_spectrum = nullptr;
        }
    }

    RfTensorShape shape;
    int samples{};
    int batches{};
    int half_bins{};
    cufftHandle forward{};
    cufftHandle inverse{};
    cufftComplex* half_spectrum{};
    float* hilbert{};
};

RfToIqPlan::RfToIqPlan(RfTensorShape shape)
    : impl_(std::make_unique<Impl>(shape)) {}

RfToIqPlan::~RfToIqPlan() = default;
RfToIqPlan::RfToIqPlan(RfToIqPlan&&) noexcept = default;
RfToIqPlan& RfToIqPlan::operator=(RfToIqPlan&&) noexcept = default;

void RfToIqPlan::transform(
    const float* device_rf,
    float2* device_iq,
    cudaStream_t stream) {
    if (device_rf == nullptr || device_iq == nullptr) {
        throw RfProcessingError("RF-to-IQ device input/output must not be null");
    }
    check_cufft(cufftSetStream(impl_->forward, stream), "set forward cuFFT stream");
    check_cufft(cufftSetStream(impl_->inverse, stream), "set inverse cuFFT stream");
    check_cufft(
        cufftExecR2C(
            impl_->forward,
            const_cast<float*>(device_rf),
            impl_->half_spectrum),
        "execute RF-to-IQ forward transform");

    constexpr unsigned int spectrum_threads = 128;
    hilbert_half_spectrum_kernel<<<
        static_cast<unsigned int>(impl_->shape.trace_count()),
        spectrum_threads,
        0,
        stream>>>(
        impl_->half_spectrum,
        impl_->samples,
        impl_->half_bins,
        impl_->shape.trace_count());
    check_cuda(cudaGetLastError(), "launch RF-to-IQ Hilbert spectrum kernel");

    check_cufft(
        cufftExecC2R(
            impl_->inverse, impl_->half_spectrum, impl_->hilbert),
        "execute RF-to-IQ inverse transform");

    constexpr unsigned int threads = 256;
    const std::size_t values = impl_->shape.value_count();
    const std::size_t blocks = (values + threads - 1) / threads;
    if (blocks > static_cast<std::size_t>(UINT_MAX)) {
        throw RfProcessingError("RF-to-IQ pack grid exceeds CUDA x dimension");
    }
    pack_analytic_iq_kernel<<<
        static_cast<unsigned int>(blocks), threads, 0, stream>>>(
        device_rf,
        impl_->hilbert,
        device_iq,
        values,
        1.0F / static_cast<float>(impl_->samples));
    check_cuda(cudaGetLastError(), "launch RF-to-IQ analytic signal pack");
}

const RfTensorShape& RfToIqPlan::shape() const noexcept {
    return impl_->shape;
}

std::size_t RfToIqPlan::workspace_bytes() const {
    const std::size_t half_values =
        impl_->shape.trace_count() * static_cast<std::size_t>(impl_->half_bins);
    return half_values * sizeof(cufftComplex) +
           impl_->shape.value_count() * sizeof(float);
}

}  // namespace carotid_doppler
