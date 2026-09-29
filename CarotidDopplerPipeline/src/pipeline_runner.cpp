#include "carotid_doppler/pipeline_runner.hpp"

#include "carotid_doppler/cuda_error.hpp"
#include "carotid_doppler/das_beamformer.hpp"
#include "carotid_doppler/global_temporal_svd.hpp"
#include "carotid_doppler/rf_bandpass.hpp"
#include "carotid_doppler/rf_reader.hpp"
#include "carotid_doppler/rf_to_iq.hpp"
#include "carotid_doppler/run_manifest.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace carotid_doppler {
namespace {

template <typename Value>
std::size_t checked_product(
    Value left,
    Value right,
    const char* description) {
    static_assert(std::is_same_v<Value, std::size_t>);
    if (left != 0 && right > std::numeric_limits<Value>::max() / left) {
        throw std::overflow_error(std::string(description) + " overflows size_t");
    }
    return left * right;
}

template <typename Value>
class DeviceBuffer final {
public:
    DeviceBuffer(std::size_t count, const char* description)
        : count_(count), description_(description) {
        if (count_ == 0) {
            throw std::invalid_argument(
                description_ + " requires a positive element count");
        }
        const std::size_t bytes = checked_product<std::size_t>(
            count_, sizeof(Value), description);
        check_cuda(
            cudaMalloc(reinterpret_cast<void**>(&pointer_), bytes),
            "allocate " + description_);
    }

    ~DeviceBuffer() {
        if (pointer_ != nullptr) {
            static_cast<void>(cudaFree(pointer_));
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    Value* get() noexcept { return pointer_; }
    const Value* get() const noexcept { return pointer_; }
    std::size_t bytes() const noexcept { return count_ * sizeof(Value); }

private:
    Value* pointer_{nullptr};
    std::size_t count_{};
    std::string description_;
};

class CudaEvent final {
public:
    explicit CudaEvent(std::string description)
        : description_(std::move(description)) {
        check_cuda(cudaEventCreate(&event_), "create " + description_);
    }

    ~CudaEvent() {
        if (event_ != nullptr) {
            static_cast<void>(cudaEventDestroy(event_));
        }
    }

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    void record(std::string_view context) {
        check_cuda(cudaEventRecord(event_), context);
    }

    cudaEvent_t get() const noexcept { return event_; }

private:
    cudaEvent_t event_{nullptr};
    std::string description_;
};

double host_elapsed_ms(
    std::chrono::steady_clock::time_point begin,
    std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

double device_elapsed_ms(
    const CudaEvent& begin,
    const CudaEvent& end,
    std::string_view context) {
    float milliseconds = 0.0F;
    check_cuda(
        cudaEventElapsedTime(&milliseconds, begin.get(), end.get()),
        context);
    return static_cast<double>(milliseconds);
}

std::string cuda_version_string(int version) {
    if (version <= 0) {
        return "unknown";
    }
    return std::to_string(version / 1000) + "." +
           std::to_string((version % 1000) / 10);
}

RuntimeSoftware query_runtime_software() {
    int runtime_version = 0;
    int driver_version = 0;
    int device = 0;
    cudaDeviceProp properties{};
    check_cuda(
        cudaRuntimeGetVersion(&runtime_version),
        "query CUDA runtime version");
    check_cuda(
        cudaDriverGetVersion(&driver_version),
        "query CUDA driver version");
    check_cuda(cudaGetDevice(&device), "query active CUDA device");
    check_cuda(
        cudaGetDeviceProperties(&properties, device),
        "query active CUDA device properties");
    return RuntimeSoftware{
        cuda_version_string(runtime_version),
        cuda_version_string(driver_version),
        properties.name};
}

void require_finite_iq(
    const Complex32* values,
    std::size_t count,
    std::size_t selected_frame_offset) {
    const auto invalid = std::find_if(
        values,
        values + count,
        [](const Complex32& value) {
            return !std::isfinite(value.real) || !std::isfinite(value.imag);
        });
    if (invalid != values + count) {
        throw std::runtime_error(
            "DAS produced non-finite beamformed IQ in batch starting at "
            "selected frame offset " +
            std::to_string(selected_frame_offset));
    }
}

std::string run_id_from_directory(const std::filesystem::path& directory) {
    std::string run_id = directory.filename().string();
    if (run_id.empty()) {
        run_id = "carotid-run";
    }
    return run_id;
}

[[noreturn]] void throw_protocol_error(
    std::string_view field,
    std::string actual,
    std::string_view expected) {
    throw std::invalid_argument(
        "formal carotid protocol violation at " + std::string(field) +
        ": got " + std::move(actual) + "; expected " +
        std::string(expected));
}

}  // namespace

void validate_real_carotid_protocol(const PipelineConfig& config) {
    const auto require_size = [](std::string_view field,
                                 std::size_t actual,
                                 std::size_t expected) {
        if (actual != expected) {
            throw_protocol_error(
                field, std::to_string(actual), std::to_string(expected));
        }
    };
    require_size("$.input.frame_start", config.input.frame_start, 0);
    require_size("$.input.frame_count", config.input.frame_count, 381);
    require_size("$.input.sequences", config.input.sequences, 5);
    require_size("$.input.channels", config.input.channels, 128);
    require_size("$.input.samples", config.input.samples, 2816);
    if (config.input.channel_order != ChannelOrder::descending_x) {
        throw_protocol_error(
            "$.input.channel_order",
            std::string(channel_order_name(config.input.channel_order)),
            "descending_x");
    }
    constexpr double expected_angles[] = {-6.0, -3.0, 0.0, 3.0, 6.0};
    if (config.acquisition.transmit_angles_deg.size() != 5) {
        throw_protocol_error(
            "$.acquisition.transmit_angles_deg",
            "array length " +
                std::to_string(config.acquisition.transmit_angles_deg.size()),
            "[-6,-3,0,3,6]");
    }
    for (std::size_t index = 0; index < 5; ++index) {
        if (config.acquisition.transmit_angles_deg[index] !=
            expected_angles[index]) {
            throw_protocol_error(
                "$.acquisition.transmit_angles_deg[" +
                    std::to_string(index) + "]",
                std::to_string(
                    config.acquisition.transmit_angles_deg[index]),
                std::to_string(expected_angles[index]));
        }
    }
    require_size(
        "$.acquisition.tx_delay_offset_samples",
        config.acquisition.tx_delay_offset_samples,
        177);
    if (config.svd.indexing != "one_based_inclusive") {
        throw_protocol_error(
            "$.svd.indexing", config.svd.indexing, "one_based_inclusive");
    }
}

void run_real_pipeline(
    const PipelineConfig& config,
    const RealPipelineOptions& options) {
    validate_real_carotid_protocol(config);
    static_assert(sizeof(float2) == sizeof(Complex32));
    static_assert(alignof(float2) == alignof(Complex32));
    if constexpr (std::endian::native != std::endian::little) {
        throw std::runtime_error(
            "real pipeline binary contract requires a little-endian host");
    }
    if (options.batch_frames == 0) {
        throw std::invalid_argument("--batch-frames must be positive");
    }
    if (options.dump_angle_beamformed_iq &&
        !options.angle_diagnostics_enabled) {
        throw std::invalid_argument(
            "dumping angle-beamformed IQ requires angle diagnostics");
    }

    const std::filesystem::path run_directory =
        std::filesystem::absolute(options.run_directory).lexically_normal();
    if (run_directory.empty()) {
        throw std::invalid_argument("real pipeline run directory is required");
    }
    if (std::filesystem::exists(run_directory)) {
        throw std::invalid_argument(
            "run directory must be new and immutable: " +
            run_directory.string());
    }

    const auto pipeline_begin = std::chrono::steady_clock::now();
    PipelineTimings timings;
    const RfFileFacts input_facts = inspect_rf_file(config.input);
    const std::vector<float> coefficients =
        load_bandpass_coefficients(config.bandpass);
    const RfTgcParameters tgc_parameters =
        make_rf_tgc_parameters(config);
    const BandpassFileIdentity bandpass_identity =
        capture_bandpass_identity(config.bandpass);

    const std::size_t pixels = checked_product<std::size_t>(
        config.image_grid.nx,
        config.image_grid.nz,
        "beamformed pixel count");
    const std::size_t beamformed_count = checked_product<std::size_t>(
        pixels,
        config.input.frame_count,
        "complete beamformed IQ count");
    std::vector<Complex32> beamformed_iq(beamformed_count);
    const std::size_t per_angle_count = checked_product<std::size_t>(
        beamformed_count,
        config.input.sequences,
        "complete per-angle IQ count");
    std::vector<Complex32> angle_beamformed_iq;
    if (options.angle_diagnostics_enabled) {
        angle_beamformed_iq.resize(per_angle_count);
    }

    std::unique_ptr<DeviceBuffer<float>> device_coefficients;
    if (config.bandpass.enabled) {
        const auto copy_begin = std::chrono::steady_clock::now();
        device_coefficients = std::make_unique<DeviceBuffer<float>>(
            coefficients.size(), "FIR coefficients");
        check_cuda(
            cudaMemcpy(
                device_coefficients->get(),
                coefficients.data(),
                device_coefficients->bytes(),
                cudaMemcpyHostToDevice),
            "copy FIR coefficients to device");
        timings.host_to_device_ms += host_elapsed_ms(
            copy_begin, std::chrono::steady_clock::now());
    }

    for (std::size_t selected_offset = 0;
         selected_offset < config.input.frame_count;) {
        const std::size_t batch_frames = std::min(
            options.batch_frames,
            config.input.frame_count - selected_offset);

        const auto read_begin = std::chrono::steady_clock::now();
        MappedRfChunk chunk = read_mapped_rf_chunk(
            config.input,
            input_facts,
            selected_offset,
            batch_frames);
        timings.read_and_map_ms += host_elapsed_ms(
            read_begin, std::chrono::steady_clock::now());

        const RfTensorShape rf_shape{
            chunk.frame_count,
            chunk.sequences,
            chunk.channels,
            chunk.samples};
        if (chunk.values.size() != rf_shape.value_count()) {
            throw std::runtime_error(
                "mapped RF batch does not match its canonical internal shape");
        }
        const DasParameters das_parameters =
            make_das_parameters(config, chunk.frame_count);

        DeviceBuffer<std::int16_t> device_rf(
            rf_shape.value_count(), "mapped RF batch");
        DeviceBuffer<float> device_tgc_rf(
            rf_shape.value_count(), "TGC RF batch");
        DeviceBuffer<float> device_filtered_rf(
            rf_shape.value_count(), "filtered RF batch");
        DeviceBuffer<float2> device_rf_iq(
            rf_shape.value_count(), "analytic RF IQ batch");
        DeviceBuffer<float2> device_beamformed(
            das_parameters.shape.output_value_count(),
            "beamformed IQ batch");
        std::unique_ptr<DeviceBuffer<float2>> device_per_angle;
        if (options.angle_diagnostics_enabled) {
            device_per_angle = std::make_unique<DeviceBuffer<float2>>(
                das_parameters.shape.per_angle_output_value_count(),
                "per-angle beamformed IQ batch");
        }

        const auto copy_begin = std::chrono::steady_clock::now();
        check_cuda(
            cudaMemcpy(
                device_rf.get(),
                chunk.values.data(),
                device_rf.bytes(),
                cudaMemcpyHostToDevice),
            "copy mapped RF batch to device");
        timings.host_to_device_ms += host_elapsed_ms(
            copy_begin, std::chrono::steady_clock::now());

        RfToIqPlan iq_plan(rf_shape);
        DasBeamformer beamformer(das_parameters);
        CudaEvent processing_start("batch processing start event");
        CudaEvent tgc_complete("batch TGC event");
        CudaEvent bandpass_complete("batch bandpass event");
        CudaEvent iq_complete("batch RF-to-IQ event");
        CudaEvent beamforming_complete("batch beamforming event");
        processing_start.record("record batch processing start");
        map_rf_and_apply_tgc(
            device_rf.get(),
            device_tgc_rf.get(),
            rf_shape,
            tgc_parameters);
        tgc_complete.record("record batch TGC completion");
        apply_centered_rf_bandpass(
            device_tgc_rf.get(),
            device_filtered_rf.get(),
            rf_shape,
            device_coefficients ? device_coefficients->get() : nullptr,
            config.bandpass.taps,
            config.bandpass.enabled);
        bandpass_complete.record("record batch bandpass completion");
        iq_plan.transform(device_filtered_rf.get(), device_rf_iq.get());
        iq_complete.record("record batch RF-to-IQ completion");
        if (device_per_angle) {
            beamformer.beamform_with_per_angle(
                device_rf_iq.get(),
                device_beamformed.get(),
                device_per_angle->get());
        } else {
            beamformer.beamform(device_rf_iq.get(), device_beamformed.get());
        }
        beamforming_complete.record("record batch beamforming completion");
        check_cuda(
            cudaEventSynchronize(beamforming_complete.get()),
            "synchronize RF-to-beamformed-IQ batch");

        timings.tgc_ms += device_elapsed_ms(
            processing_start,
            tgc_complete,
            "measure batch TGC time");
        timings.bandpass_ms += device_elapsed_ms(
            tgc_complete,
            bandpass_complete,
            "measure batch bandpass time");
        timings.rf_to_iq_ms += device_elapsed_ms(
            bandpass_complete,
            iq_complete,
            "measure batch RF-to-IQ time");
        timings.beamforming_ms += device_elapsed_ms(
            iq_complete,
            beamforming_complete,
            "measure batch beamforming time");

        const std::size_t output_count =
            das_parameters.shape.output_value_count();
        Complex32* const host_destination =
            beamformed_iq.data() + selected_offset * pixels;
        const auto copy_back_begin = std::chrono::steady_clock::now();
        check_cuda(
            cudaMemcpy(
                host_destination,
                device_beamformed.get(),
                checked_product<std::size_t>(
                    output_count,
                    sizeof(Complex32),
                    "beamformed batch byte count"),
                cudaMemcpyDeviceToHost),
            "copy beamformed IQ batch to host cache");
        timings.beamformed_to_host_ms += host_elapsed_ms(
            copy_back_begin, std::chrono::steady_clock::now());
        require_finite_iq(host_destination, output_count, selected_offset);
        if (device_per_angle) {
            const std::size_t per_angle_batch_count =
                das_parameters.shape.per_angle_output_value_count();
            Complex32* const per_angle_destination =
                angle_beamformed_iq.data() +
                selected_offset * config.input.sequences * pixels;
            const auto per_angle_copy_begin = std::chrono::steady_clock::now();
            check_cuda(
                cudaMemcpy(
                    per_angle_destination,
                    device_per_angle->get(),
                    checked_product<std::size_t>(
                        per_angle_batch_count,
                        sizeof(Complex32),
                        "per-angle batch byte count"),
                    cudaMemcpyDeviceToHost),
                "copy per-angle beamformed IQ batch to host cache");
            timings.beamformed_to_host_ms += host_elapsed_ms(
                per_angle_copy_begin, std::chrono::steady_clock::now());
            require_finite_iq(
                per_angle_destination,
                per_angle_batch_count,
                selected_offset);
        }
        selected_offset += batch_frames;
    }

    const GlobalTemporalSvdConfig svd_config{
        pixels,
        config.input.frame_count,
        config.svd.keep_start,
        config.svd.keep_end};
    const auto svd_begin = std::chrono::steady_clock::now();
    GlobalTemporalSvdResult svd_result = global_temporal_svd_host(
        beamformed_iq.data(),
        beamformed_iq.size(),
        svd_config,
        options.dump_filtered_iq);
    const double svd_host_total_ms = host_elapsed_ms(
        svd_begin, std::chrono::steady_clock::now());
    timings.global_svd = svd_result.timings;
    timings.svd_host_overhead_ms = std::max(
        0.0, svd_host_total_ms - timings.global_svd.total_ms);

    std::vector<float> angle_power_linear;
    std::vector<float> angle_singular_values;
    std::vector<float> incoherent_power_linear;
    std::vector<float> angle_coherence;
    if (options.angle_diagnostics_enabled) {
        angle_power_linear.resize(checked_product<std::size_t>(
            config.input.sequences, pixels, "angle power value count"));
        angle_singular_values.resize(checked_product<std::size_t>(
            config.input.sequences,
            config.input.frame_count,
            "angle singular-value count"));
        incoherent_power_linear.assign(pixels, 0.0F);
        angle_coherence.resize(pixels);
        for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
            double numerator = 0.0;
            double denominator = 0.0;
            for (std::size_t frame = 0;
                 frame < config.input.frame_count;
                 ++frame) {
                double sum_real = 0.0;
                double sum_imag = 0.0;
                double sum_angle_energy = 0.0;
                for (std::size_t angle = 0;
                     angle < config.input.sequences;
                     ++angle) {
                    const Complex32& value = angle_beamformed_iq[
                        (frame * config.input.sequences + angle) * pixels +
                        pixel];
                    sum_real += value.real;
                    sum_imag += value.imag;
                    sum_angle_energy +=
                        static_cast<double>(value.real) * value.real +
                        static_cast<double>(value.imag) * value.imag;
                }
                numerator += sum_real * sum_real + sum_imag * sum_imag;
                denominator +=
                    static_cast<double>(config.input.sequences) *
                    sum_angle_energy;
            }
            angle_coherence[pixel] = denominator > 0.0
                ? static_cast<float>(numerator / denominator)
                : 0.0F;
        }
        std::vector<Complex32> one_angle_iq(beamformed_count);
        const auto angle_svd_begin = std::chrono::steady_clock::now();
        for (std::size_t angle = 0; angle < config.input.sequences; ++angle) {
            for (std::size_t frame = 0; frame < config.input.frame_count; ++frame) {
                const Complex32* const source =
                    angle_beamformed_iq.data() +
                    (frame * config.input.sequences + angle) * pixels;
                std::copy_n(
                    source,
                    pixels,
                    one_angle_iq.data() + frame * pixels);
            }
            GlobalTemporalSvdResult angle_result = global_temporal_svd_host(
                one_angle_iq.data(),
                one_angle_iq.size(),
                svd_config,
                false);
            std::copy(
                angle_result.power_linear.begin(),
                angle_result.power_linear.end(),
                angle_power_linear.begin() + angle * pixels);
            for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
                incoherent_power_linear[pixel] +=
                    angle_result.power_linear[pixel];
            }
            std::copy(
                angle_result.singular_values.begin(),
                angle_result.singular_values.end(),
                angle_singular_values.begin() +
                    angle * config.input.frame_count);
        }
        timings.angle_svd_total_ms = host_elapsed_ms(
            angle_svd_begin, std::chrono::steady_clock::now());
    }

    RealRunArtifacts artifacts;
    artifacts.power_linear = std::move(svd_result.power_linear);
    artifacts.singular_values = std::move(svd_result.singular_values);
    if (options.dump_beamformed_iq) {
        artifacts.beamformed_iq = std::move(beamformed_iq);
    }
    if (options.angle_diagnostics_enabled) {
        if (options.dump_angle_beamformed_iq) {
            artifacts.angle_beamformed_iq = std::move(angle_beamformed_iq);
        }
        artifacts.angle_power_linear = std::move(angle_power_linear);
        artifacts.angle_singular_values = std::move(angle_singular_values);
        artifacts.incoherent_power_linear =
            std::move(incoherent_power_linear);
        artifacts.angle_coherence = std::move(angle_coherence);
    }
    if (options.dump_filtered_iq) {
        artifacts.filtered_iq = std::move(svd_result.filtered_iq);
    }
    artifacts.software = query_runtime_software();
    timings.total_ms = host_elapsed_ms(
        pipeline_begin, std::chrono::steady_clock::now());
    artifacts.timings = timings;

    write_complete_real_run(
        RealRunSpec{
            run_directory,
            run_id_from_directory(run_directory),
            config,
            input_facts,
            bandpass_identity,
            options.batch_frames,
            options.dump_beamformed_iq,
            options.angle_diagnostics_enabled,
            options.dump_angle_beamformed_iq,
            options.dump_filtered_iq},
        artifacts);
}

}  // namespace carotid_doppler
