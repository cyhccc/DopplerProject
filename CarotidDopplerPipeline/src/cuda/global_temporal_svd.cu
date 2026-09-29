#include "carotid_doppler/global_temporal_svd.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace carotid_doppler {
namespace {

static_assert(sizeof(Complex32) == sizeof(cuFloatComplex));
static_assert(alignof(Complex32) == alignof(cuFloatComplex));
static_assert(std::is_trivially_copyable<Complex32>::value);

std::string dimensions(const GlobalTemporalSvdConfig& config) {
    std::ostringstream stream;
    stream << "pixels=" << config.pixels << ", frames=" << config.frames
           << ", keep=[" << config.keep_start << ',' << config.keep_end << ']';
    return stream.str();
}

void check_cuda(
    cudaError_t status,
    const char* operation,
    const GlobalTemporalSvdConfig& config) {
    if (status == cudaSuccess) {
        return;
    }
    std::ostringstream message;
    message << "global temporal SVD " << operation << " failed ("
            << dimensions(config) << "): " << cudaGetErrorString(status);
    throw std::runtime_error(message.str());
}

const char* cublas_status_name(cublasStatus_t status) {
    switch (status) {
    case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
    default: return "CUBLAS_STATUS_UNKNOWN";
    }
}

void check_cublas(
    cublasStatus_t status,
    const char* operation,
    const GlobalTemporalSvdConfig& config) {
    if (status == CUBLAS_STATUS_SUCCESS) {
        return;
    }
    std::ostringstream message;
    message << "global temporal SVD " << operation << " failed ("
            << dimensions(config) << "): " << cublas_status_name(status);
    throw std::runtime_error(message.str());
}

const char* cusolver_status_name(cusolverStatus_t status) {
    switch (status) {
    case CUSOLVER_STATUS_SUCCESS: return "CUSOLVER_STATUS_SUCCESS";
    case CUSOLVER_STATUS_NOT_INITIALIZED: return "CUSOLVER_STATUS_NOT_INITIALIZED";
    case CUSOLVER_STATUS_ALLOC_FAILED: return "CUSOLVER_STATUS_ALLOC_FAILED";
    case CUSOLVER_STATUS_INVALID_VALUE: return "CUSOLVER_STATUS_INVALID_VALUE";
    case CUSOLVER_STATUS_ARCH_MISMATCH: return "CUSOLVER_STATUS_ARCH_MISMATCH";
    case CUSOLVER_STATUS_EXECUTION_FAILED: return "CUSOLVER_STATUS_EXECUTION_FAILED";
    case CUSOLVER_STATUS_INTERNAL_ERROR: return "CUSOLVER_STATUS_INTERNAL_ERROR";
    case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
        return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    case CUSOLVER_STATUS_NOT_SUPPORTED: return "CUSOLVER_STATUS_NOT_SUPPORTED";
    case CUSOLVER_STATUS_ZERO_PIVOT: return "CUSOLVER_STATUS_ZERO_PIVOT";
    case CUSOLVER_STATUS_INVALID_LICENSE: return "CUSOLVER_STATUS_INVALID_LICENSE";
    default: return "CUSOLVER_STATUS_UNKNOWN";
    }
}

void check_cusolver(
    cusolverStatus_t status,
    const char* operation,
    const GlobalTemporalSvdConfig& config) {
    if (status == CUSOLVER_STATUS_SUCCESS) {
        return;
    }
    std::ostringstream message;
    message << "global temporal SVD " << operation << " failed ("
            << dimensions(config) << "): " << cusolver_status_name(status);
    throw std::runtime_error(message.str());
}

std::size_t checked_product(
    std::size_t left,
    std::size_t right,
    const char* name) {
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::invalid_argument(std::string(name) + " overflows size_t");
    }
    return left * right;
}

int checked_blas_dimension(std::size_t value, const char* name) {
    if (value > static_cast<std::size_t>(INT_MAX)) {
        throw std::invalid_argument(std::string(name) + " exceeds cuBLAS int range");
    }
    return static_cast<int>(value);
}

void validate_config(const GlobalTemporalSvdConfig& config) {
    if (config.pixels == 0 || config.frames == 0) {
        throw std::invalid_argument("global temporal SVD requires pixels>=1 and frames>=1");
    }
    if (config.keep_start == 0 || config.keep_start > config.keep_end ||
        config.keep_end > config.frames) {
        std::ostringstream message;
        message << "global temporal SVD keep range must satisfy "
                   "1 <= keep_start <= keep_end <= frames; got "
                << dimensions(config);
        throw std::invalid_argument(message.str());
    }
    checked_blas_dimension(config.pixels, "pixels");
    checked_blas_dimension(config.frames, "frames");
    checked_product(config.pixels, config.frames, "pixels*frames");
    checked_product(config.frames, config.frames, "frames*frames");
}

template <typename T>
class DeviceBuffer final {
public:
    DeviceBuffer(
        std::size_t count,
        const char* name,
        const GlobalTemporalSvdConfig& config)
    {
        const std::size_t bytes = checked_product(count, sizeof(T), name);
        check_cuda(
            cudaMalloc(reinterpret_cast<void**>(&pointer_), bytes),
            name,
            config);
    }

    ~DeviceBuffer() {
        if (pointer_ != nullptr) {
            static_cast<void>(cudaFree(pointer_));
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() noexcept { return pointer_; }
    const T* get() const noexcept { return pointer_; }

private:
    T* pointer_{nullptr};
};

class CublasHandle final {
public:
    explicit CublasHandle(const GlobalTemporalSvdConfig& config) {
        check_cublas(cublasCreate(&handle_), "create cuBLAS handle", config);
    }

    ~CublasHandle() {
        if (handle_ != nullptr) {
            static_cast<void>(cublasDestroy(handle_));
        }
    }

    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;

    cublasHandle_t get() const noexcept { return handle_; }

private:
    cublasHandle_t handle_{nullptr};
};

class CusolverHandle final {
public:
    explicit CusolverHandle(const GlobalTemporalSvdConfig& config) {
        check_cusolver(cusolverDnCreate(&handle_), "create cuSOLVER handle", config);
    }

    ~CusolverHandle() {
        if (handle_ != nullptr) {
            static_cast<void>(cusolverDnDestroy(handle_));
        }
    }

    CusolverHandle(const CusolverHandle&) = delete;
    CusolverHandle& operator=(const CusolverHandle&) = delete;

    cusolverDnHandle_t get() const noexcept { return handle_; }

private:
    cusolverDnHandle_t handle_{nullptr};
};

class CudaEvent final {
public:
    explicit CudaEvent(const GlobalTemporalSvdConfig& config)
        : config_(config) {
        check_cuda(cudaEventCreate(&event_), "create timing event", config_);
    }

    ~CudaEvent() {
        if (event_ != nullptr) {
            static_cast<void>(cudaEventDestroy(event_));
        }
    }

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    void record(const char* operation) {
        check_cuda(cudaEventRecord(event_), operation, config_);
    }

    cudaEvent_t get() const noexcept { return event_; }

private:
    GlobalTemporalSvdConfig config_;
    cudaEvent_t event_{nullptr};
};

double elapsed_ms(
    const CudaEvent& begin,
    const CudaEvent& end,
    const GlobalTemporalSvdConfig& config,
    const char* operation) {
    float milliseconds = 0.0F;
    check_cuda(
        cudaEventElapsedTime(&milliseconds, begin.get(), end.get()),
        operation,
        config);
    return static_cast<double>(milliseconds);
}

__global__ void descending_singular_values_kernel(
    const float* eigen_values,
    float* singular_values,
    int frames) {
    const int component = blockIdx.x * blockDim.x + threadIdx.x;
    if (component >= frames) {
        return;
    }
    const float eigen_value = eigen_values[frames - 1 - component];
    singular_values[component] = sqrtf(fmaxf(eigen_value, 0.0F));
}

__global__ void projected_power_kernel(
    const cuFloatComplex* projected,
    float* power,
    int pixels,
    int kept_components) {
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixels) {
        return;
    }

    double sum = 0.0;
    for (int component = 0; component < kept_components; ++component) {
        const cuFloatComplex value =
            projected[pixel + static_cast<std::size_t>(component) * pixels];
        sum += static_cast<double>(value.x) * static_cast<double>(value.x) +
               static_cast<double>(value.y) * static_cast<double>(value.y);
    }
    power[pixel] = static_cast<float>(sum);
}

}  // namespace

void global_temporal_svd_device(
    const Complex32* d_input_iq,
    Complex32* d_filtered_iq,
    float* d_singular_values,
    float* d_power_linear,
    const GlobalTemporalSvdConfig& config,
    GlobalTemporalSvdTimings* timings) {
    validate_config(config);
    if (d_input_iq == nullptr || d_singular_values == nullptr ||
        d_power_linear == nullptr) {
        throw std::invalid_argument(
            "global temporal SVD requires non-null input, singular-value, and power pointers");
    }

    const int pixels = checked_blas_dimension(config.pixels, "pixels");
    const int frames = checked_blas_dimension(config.frames, "frames");
    const std::size_t kept_size = config.keep_end - config.keep_start + 1;
    const int kept_components =
        checked_blas_dimension(kept_size, "kept components");
    checked_product(config.pixels, config.frames, "pixels*frames");
    const std::size_t covariance_elements =
        checked_product(config.frames, config.frames, "frames*frames");
    const std::size_t projected_elements =
        checked_product(config.pixels, kept_size, "pixels*kept components");

    CublasHandle cublas(config);
    CusolverHandle cusolver(config);
    DeviceBuffer<cuFloatComplex> covariance(
        covariance_elements, "allocate temporal covariance", config);
    DeviceBuffer<float> eigen_values(
        config.frames, "allocate covariance eigenvalues", config);
    DeviceBuffer<cuFloatComplex> projected(
        projected_elements, "allocate retained temporal projection", config);
    DeviceBuffer<int> solver_info(1, "allocate cuSOLVER info", config);

    CudaEvent start(config);
    CudaEvent covariance_complete(config);
    CudaEvent eigensolve_complete(config);
    CudaEvent projection_complete(config);
    CudaEvent power_complete(config);
    CudaEvent reconstruction_complete(config);
    start.record("record SVD start");

    const auto* input = reinterpret_cast<const cuFloatComplex*>(d_input_iq);
    auto* filtered = reinterpret_cast<cuFloatComplex*>(d_filtered_iq);
    constexpr float alpha_hermitian = 1.0F;
    constexpr float beta_hermitian = 0.0F;
    check_cublas(
        cublasCherk(
            cublas.get(),
            CUBLAS_FILL_MODE_LOWER,
            CUBLAS_OP_C,
            frames,
            pixels,
            &alpha_hermitian,
            input,
            pixels,
            &beta_hermitian,
            covariance.get(),
            frames),
        "compute A^H*A temporal covariance",
        config);
    covariance_complete.record("record covariance completion");

    int workspace_count = 0;
    check_cusolver(
        cusolverDnCheevd_bufferSize(
            cusolver.get(),
            CUSOLVER_EIG_MODE_VECTOR,
            CUBLAS_FILL_MODE_LOWER,
            frames,
            covariance.get(),
            frames,
            eigen_values.get(),
            &workspace_count),
        "query Hermitian eigensolver workspace",
        config);
    if (workspace_count <= 0) {
        throw std::runtime_error(
            "global temporal SVD cuSOLVER returned an invalid workspace size (" +
            dimensions(config) + ")");
    }
    DeviceBuffer<cuFloatComplex> solver_workspace(
        static_cast<std::size_t>(workspace_count),
        "allocate Hermitian eigensolver workspace",
        config);
    check_cusolver(
        cusolverDnCheevd(
            cusolver.get(),
            CUSOLVER_EIG_MODE_VECTOR,
            CUBLAS_FILL_MODE_LOWER,
            frames,
            covariance.get(),
            frames,
            eigen_values.get(),
            solver_workspace.get(),
            workspace_count,
            solver_info.get()),
        "solve Hermitian temporal covariance",
        config);

    int solver_info_host = 0;
    check_cuda(
        cudaMemcpy(
            &solver_info_host,
            solver_info.get(),
            sizeof(solver_info_host),
            cudaMemcpyDeviceToHost),
        "copy cuSOLVER info",
        config);
    if (solver_info_host != 0) {
        std::ostringstream message;
        message << "global temporal SVD Hermitian eigensolver returned info="
                << solver_info_host << " (" << dimensions(config) << ')';
        throw std::runtime_error(message.str());
    }

    constexpr int threads = 256;
    descending_singular_values_kernel<<<(frames + threads - 1) / threads, threads>>>(
        eigen_values.get(), d_singular_values, frames);
    check_cuda(cudaGetLastError(), "launch singular-value conversion", config);
    eigensolve_complete.record("record eigensolve completion");

    // cuSOLVER stores eigenvectors in ascending eigenvalue order. The 1-based
    // descending keep band [start,end] is therefore the contiguous ascending
    // column interval [frames-end, frames-start]. Column order is irrelevant to
    // the orthogonal projector V_keep*V_keep^H.
    const std::size_t first_kept_ascending = config.frames - config.keep_end;
    const cuFloatComplex* retained_vectors =
        covariance.get() + first_kept_ascending * config.frames;
    const cuFloatComplex one = make_cuFloatComplex(1.0F, 0.0F);
    const cuFloatComplex zero = make_cuFloatComplex(0.0F, 0.0F);
    check_cublas(
        cublasCgemm(
            cublas.get(),
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            pixels,
            kept_components,
            frames,
            &one,
            input,
            pixels,
            retained_vectors,
            frames,
            &zero,
            projected.get(),
            pixels),
        "project IQ onto retained temporal modes",
        config);
    projection_complete.record("record temporal projection completion");

    projected_power_kernel<<<(pixels + threads - 1) / threads, threads>>>(
        projected.get(), d_power_linear, pixels, kept_components);
    check_cuda(cudaGetLastError(), "launch retained-mode power accumulation", config);
    power_complete.record("record power completion");

    if (filtered != nullptr) {
        check_cublas(
            cublasCgemm(
                cublas.get(),
                CUBLAS_OP_N,
                CUBLAS_OP_C,
                pixels,
                frames,
                kept_components,
                &one,
                projected.get(),
                pixels,
                retained_vectors,
                frames,
                &zero,
                filtered,
                pixels),
            "reconstruct diagnostic filtered IQ",
            config);
    }

    reconstruction_complete.record("record filtered reconstruction completion");

    check_cuda(
        cudaEventSynchronize(reconstruction_complete.get()),
        "synchronize completed SVD outputs",
        config);
    if (timings != nullptr) {
        timings->covariance_ms = elapsed_ms(
            start, covariance_complete, config, "measure covariance time");
        timings->eigensolve_ms = elapsed_ms(
            covariance_complete,
            eigensolve_complete,
            config,
            "measure eigensolve time");
        timings->projection_ms = elapsed_ms(
            eigensolve_complete,
            projection_complete,
            config,
            "measure temporal projection time");
        timings->power_ms = elapsed_ms(
            projection_complete,
            power_complete,
            config,
            "measure power time");
        timings->filtered_reconstruction_ms = d_filtered_iq == nullptr
            ? 0.0
            : elapsed_ms(
                  power_complete,
                  reconstruction_complete,
                  config,
                  "measure filtered reconstruction time");
        timings->total_ms = elapsed_ms(
            start,
            reconstruction_complete,
            config,
            "measure total SVD time");
    }
}

GlobalTemporalSvdResult global_temporal_svd_host(
    const Complex32* input_iq,
    std::size_t input_count,
    const GlobalTemporalSvdConfig& config,
    bool include_filtered_iq) {
    validate_config(config);
    const std::size_t matrix_elements =
        checked_product(config.pixels, config.frames, "pixels*frames");
    if (input_iq == nullptr) {
        throw std::invalid_argument("global temporal SVD host input pointer is null");
    }
    if (input_count != matrix_elements) {
        std::ostringstream message;
        message << "global temporal SVD host input count mismatch: expected "
                << matrix_elements << ", actual " << input_count;
        throw std::invalid_argument(message.str());
    }

    DeviceBuffer<Complex32> device_input(
        matrix_elements, "allocate host-seam input IQ", config);
    DeviceBuffer<float> device_singular_values(
        config.frames, "allocate host-seam singular values", config);
    DeviceBuffer<float> device_power(
        config.pixels, "allocate host-seam linear power", config);
    check_cuda(
        cudaMemcpy(
            device_input.get(),
            input_iq,
            checked_product(matrix_elements, sizeof(Complex32), "input IQ bytes"),
            cudaMemcpyHostToDevice),
        "copy host IQ to device",
        config);

    GlobalTemporalSvdResult result;
    result.singular_values.resize(config.frames);
    result.power_linear.resize(config.pixels);

    Complex32* filtered_pointer = nullptr;
    if (include_filtered_iq) {
        result.filtered_iq.resize(matrix_elements);
        DeviceBuffer<Complex32> device_filtered(
            matrix_elements, "allocate host-seam filtered IQ", config);
        filtered_pointer = device_filtered.get();
        global_temporal_svd_device(
            device_input.get(),
            filtered_pointer,
            device_singular_values.get(),
            device_power.get(),
            config,
            &result.timings);
        check_cuda(
            cudaMemcpy(
                result.filtered_iq.data(),
                device_filtered.get(),
                checked_product(
                    matrix_elements, sizeof(Complex32), "filtered IQ bytes"),
                cudaMemcpyDeviceToHost),
            "copy filtered IQ to host",
            config);
    } else {
        global_temporal_svd_device(
            device_input.get(),
            nullptr,
            device_singular_values.get(),
            device_power.get(),
            config,
            &result.timings);
    }

    check_cuda(
        cudaMemcpy(
            result.singular_values.data(),
            device_singular_values.get(),
            checked_product(config.frames, sizeof(float), "singular-value bytes"),
            cudaMemcpyDeviceToHost),
        "copy singular values to host",
        config);
    check_cuda(
        cudaMemcpy(
            result.power_linear.data(),
            device_power.get(),
            checked_product(config.pixels, sizeof(float), "power bytes"),
            cudaMemcpyDeviceToHost),
        "copy linear power to host",
        config);
    return result;
}

}  // namespace carotid_doppler
