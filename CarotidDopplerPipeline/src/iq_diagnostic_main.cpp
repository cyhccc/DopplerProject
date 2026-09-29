#include "carotid_doppler/config.hpp"
#include "carotid_doppler/cuda_error.hpp"
#include "carotid_doppler/rf_bandpass.hpp"
#include "carotid_doppler/rf_reader.hpp"
#include "carotid_doppler/rf_to_iq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace {

struct CliOptions {
    bool show_help{false};
    std::filesystem::path config_path;
    std::filesystem::path output_directory;
    std::size_t selected_frame_offset{0};
    std::size_t frame_count{1};
};

void print_usage(std::ostream& output) {
    output
        << "Usage:\n"
        << "  carotid_iq_diagnostic --config <pipeline.json> "
           "--output-dir <new-directory> [--frame-offset <zero-based>] "
           "[--frame-count <positive>]\n\n"
        << "This opt-in diagnostic reads mapped RF in "
           "[frame,sequence,internal_channel,sample] order, applies the "
           "configured TGC and 129-tap FIR, and writes analytic complex float32 IQ. "
           "It does not perform DAS, SVD, power, dB conversion, or rendering.\n";
}

std::size_t parse_size(
    std::string_view text,
    std::string_view option,
    bool allow_zero) {
    std::size_t value = 0;
    const auto result =
        std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size() ||
        (!allow_zero && value == 0)) {
        throw std::invalid_argument(
            std::string(option) +
            (allow_zero ? " expects a non-negative integer" :
                          " expects a positive integer"));
    }
    return value;
}

CliOptions parse_cli(int argc, char** argv) {
    CliOptions options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--help" || argument == "-h") {
            options.show_help = true;
            continue;
        }
        if (argument != "--config" && argument != "--output-dir" &&
            argument != "--frame-offset" && argument != "--frame-count") {
            throw std::invalid_argument(
                "unknown option: " + std::string(argument));
        }
        if (index + 1 >= argc) {
            throw std::invalid_argument(
                std::string(argument) + " requires a value");
        }
        const std::string_view value(argv[++index]);
        if (argument == "--config") {
            options.config_path = std::filesystem::path(value);
        } else if (argument == "--output-dir") {
            options.output_directory = std::filesystem::path(value);
        } else if (argument == "--frame-offset") {
            options.selected_frame_offset =
                parse_size(value, argument, true);
        } else {
            options.frame_count = parse_size(value, argument, false);
        }
    }
    if (options.show_help) {
        return options;
    }
    if (options.config_path.empty()) {
        throw std::invalid_argument("--config is required");
    }
    if (options.output_directory.empty()) {
        throw std::invalid_argument("--output-dir is required");
    }
    return options;
}

template <typename Value>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        if (count == 0 || count >
                              std::numeric_limits<std::size_t>::max() /
                                  sizeof(Value)) {
            throw std::runtime_error("invalid CUDA device allocation size");
        }
        carotid_doppler::check_cuda(
            cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(Value)),
            "allocate IQ diagnostic device buffer");
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    Value* get() noexcept { return data_; }
    const Value* get() const noexcept { return data_; }
    std::size_t bytes() const noexcept { return count_ * sizeof(Value); }

private:
    Value* data_{};
    std::size_t count_{};
};

class CudaEvent {
public:
    CudaEvent() {
        carotid_doppler::check_cuda(
            cudaEventCreate(&event_), "create IQ diagnostic CUDA event");
    }
    ~CudaEvent() {
        if (event_ != nullptr) {
            cudaEventDestroy(event_);
        }
    }
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    cudaEvent_t get() const noexcept { return event_; }

private:
    cudaEvent_t event_{};
};

std::string json_escape(std::string_view value) {
    std::ostringstream escaped;
    for (const char character : value) {
        switch (character) {
        case '\\': escaped << "\\\\"; break;
        case '"': escaped << "\\\""; break;
        case '\n': escaped << "\\n"; break;
        case '\r': escaped << "\\r"; break;
        case '\t': escaped << "\\t"; break;
        default: escaped << character; break;
        }
    }
    return escaped.str();
}

void write_atomic_bytes(
    const std::filesystem::path& target,
    const void* values,
    std::size_t byte_count) {
    if (byte_count > static_cast<std::size_t>(
                         std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("artifact exceeds std::streamsize range");
    }
    const std::filesystem::path temporary =
        target.parent_path() / (target.filename().string() + ".tmp");
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    try {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) {
            throw std::runtime_error(
                "failed to create temporary artifact: " + temporary.string());
        }
        output.write(
            reinterpret_cast<const char*>(values),
            static_cast<std::streamsize>(byte_count));
        output.flush();
        if (!output) {
            throw std::runtime_error(
                "failed while writing temporary artifact: " +
                temporary.string());
        }
        output.close();
        if (std::filesystem::file_size(temporary) != byte_count) {
            throw std::runtime_error("temporary artifact byte count mismatch");
        }
        std::filesystem::rename(temporary, target);
    } catch (...) {
        std::filesystem::remove(temporary, ignored);
        throw;
    }
}

void write_atomic_text(
    const std::filesystem::path& target,
    const std::string& text) {
    write_atomic_bytes(target, text.data(), text.size());
}

double elapsed_ms(
    std::chrono::steady_clock::time_point begin,
    std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

std::string make_manifest(
    const carotid_doppler::PipelineConfig& config,
    const carotid_doppler::MappedRfChunk& chunk,
    const std::filesystem::path& iq_path,
    std::size_t iq_bytes,
    std::size_t workspace_bytes,
    double read_ms,
    float tgc_ms,
    float bandpass_ms,
    float rf_to_iq_ms,
    double write_ms,
    float maximum_magnitude) {
    std::ostringstream output;
    output << std::setprecision(10)
           << "{\n"
           << "  \"schema_version\": 1,\n"
           << "  \"kind\": \"rf_iq_diagnostic\",\n"
           << "  \"status\": \"complete\",\n"
           << "  \"config_path\": \""
           << json_escape(config.source_path.generic_string()) << "\",\n"
           << "  \"input\": {\n"
           << "    \"path\": \""
           << json_escape(config.input.path.generic_string()) << "\",\n"
           << "    \"dtype\": \"int16_le\",\n"
           << "    \"source_frame_start\": " << chunk.source_frame_start
           << ",\n"
           << "    \"shape\": [" << chunk.frame_count << ", "
           << chunk.sequences << ", " << chunk.channels << ", "
           << chunk.samples << "],\n"
           << "    \"layout\": [\"frame\", \"sequence\", "
              "\"internal_channel\", \"sample\"],\n"
           << "    \"internal_channel_order\": \"ascending_x\"\n"
           << "  },\n"
           << "  \"processing\": {\n"
           << "    \"tgc_enabled\": "
           << (config.tgc.enabled ? "true" : "false") << ",\n"
           << "    \"tgc_alpha_coeff_db_cm_mhz_power\": "
           << config.tgc.alpha_coeff_db_cm_mhz_power << ",\n"
           << "    \"tgc_frequency_power\": "
           << config.tgc.frequency_power << ",\n"
           << "    \"transmit_cycle_count\": "
           << config.acquisition.transmit_cycle_count << ",\n"
           << "    \"preprocessing_order\": \"int16_map,tgc,centered_bandpass,hilbert\",\n"
           << "    \"bandpass_enabled\": "
           << (config.bandpass.enabled ? "true" : "false") << ",\n"
           << "    \"bandpass_coefficients_path\": \""
           << json_escape(config.bandpass.coefficients_path.generic_string())
           << "\",\n"
           << "    \"bandpass_dtype\": \"float32_le\",\n"
           << "    \"bandpass_taps\": 129,\n"
           << "    \"bandpass_alignment\": \"centered_zero_padded\",\n"
           << "    \"iq_method\": \"fft_discrete_hilbert\",\n"
           << "    \"iq_phase_convention\": "
              "\"real=filtered_rf, imag=Hilbert(filtered_rf)\",\n"
           << "    \"workspace_bytes\": " << workspace_bytes << "\n"
           << "  },\n"
           << "  \"artifact\": {\n"
           << "    \"path\": \"" << json_escape(iq_path.filename().string())
           << "\",\n"
           << "    \"scalar_dtype\": \"float32_le\",\n"
           << "    \"complex_encoding\": \"interleaved_real_imag\",\n"
           << "    \"shape\": [" << chunk.frame_count << ", "
           << chunk.sequences << ", " << chunk.channels << ", "
           << chunk.samples << "],\n"
           << "    \"layout\": [\"frame\", \"sequence\", "
              "\"internal_channel\", \"sample\"],\n"
           << "    \"contiguous_axis\": \"sample\",\n"
           << "    \"complex_index\": "
              "\"(((frame*sequences+sequence)*channels+channel)*samples)+sample\",\n"
           << "    \"real_offset\": \"2*complex_index\",\n"
           << "    \"imag_offset\": \"2*complex_index+1\",\n"
           << "    \"size_bytes\": " << iq_bytes << ",\n"
           << "    \"maximum_magnitude\": " << maximum_magnitude << "\n"
           << "  },\n"
           << "  \"timings_ms\": {\n"
           << "    \"read_and_map\": " << read_ms << ",\n"
           << "    \"tgc_or_map\": " << tgc_ms << ",\n"
           << "    \"bandpass_or_copy\": " << bandpass_ms << ",\n"
           << "    \"rf_to_iq\": " << rf_to_iq_ms << ",\n"
           << "    \"write\": " << write_ms << "\n"
           << "  }\n"
           << "}\n";
    return output.str();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        static_assert(sizeof(float2) == 2 * sizeof(float));
        if constexpr (std::endian::native != std::endian::little) {
            throw std::runtime_error(
                "IQ diagnostic binary contract requires a little-endian host");
        }

        const CliOptions options = parse_cli(argc, argv);
        if (options.show_help) {
            print_usage(std::cout);
            return 0;
        }
        const auto config =
            carotid_doppler::load_pipeline_config(options.config_path);
        if (options.selected_frame_offset > config.input.frame_count ||
            options.frame_count >
                config.input.frame_count - options.selected_frame_offset) {
            throw std::invalid_argument(
                "diagnostic frame window exceeds the configured RF selection");
        }

        const std::filesystem::path output_directory =
            std::filesystem::absolute(options.output_directory)
                .lexically_normal();
        if (std::filesystem::exists(output_directory)) {
            throw std::runtime_error(
                "diagnostic output directory must not already exist: " +
                output_directory.string());
        }
        std::filesystem::create_directories(output_directory);

        const auto read_begin = std::chrono::steady_clock::now();
        const auto facts = carotid_doppler::inspect_rf_file(config.input);
        const auto chunk = carotid_doppler::read_mapped_rf_chunk(
            config.input,
            facts,
            options.selected_frame_offset,
            options.frame_count);
        const auto coefficients =
            carotid_doppler::load_bandpass_coefficients(config.bandpass);
        const auto read_end = std::chrono::steady_clock::now();

        const carotid_doppler::RfTensorShape shape{
            chunk.frame_count,
            chunk.sequences,
            chunk.channels,
            chunk.samples};
        if (chunk.values.size() != shape.value_count()) {
            throw std::runtime_error(
                "mapped RF chunk does not match declared internal shape");
        }

        DeviceBuffer<std::int16_t> device_rf(shape.value_count());
        DeviceBuffer<float> device_tgc(shape.value_count());
        DeviceBuffer<float> device_filtered(shape.value_count());
        DeviceBuffer<float2> device_iq(shape.value_count());
        carotid_doppler::check_cuda(
            cudaMemcpy(
                device_rf.get(),
                chunk.values.data(),
                device_rf.bytes(),
                cudaMemcpyHostToDevice),
            "copy mapped RF to device");

        std::unique_ptr<DeviceBuffer<float>> device_coefficients;
        if (config.bandpass.enabled) {
            device_coefficients =
                std::make_unique<DeviceBuffer<float>>(coefficients.size());
            carotid_doppler::check_cuda(
                cudaMemcpy(
                    device_coefficients->get(),
                    coefficients.data(),
                    device_coefficients->bytes(),
                    cudaMemcpyHostToDevice),
                "copy FIR coefficients to device");
        }

        carotid_doppler::RfToIqPlan iq_plan(shape);
        CudaEvent start;
        CudaEvent tgc_complete;
        CudaEvent bandpass_complete;
        CudaEvent iq_complete;
        carotid_doppler::check_cuda(
            cudaEventRecord(start.get()), "record RF processing start");
        carotid_doppler::map_rf_and_apply_tgc(
            device_rf.get(),
            device_tgc.get(),
            shape,
            carotid_doppler::make_rf_tgc_parameters(config));
        carotid_doppler::check_cuda(
            cudaEventRecord(tgc_complete.get()), "record TGC completion");
        carotid_doppler::apply_centered_rf_bandpass(
            device_tgc.get(),
            device_filtered.get(),
            shape,
            device_coefficients ? device_coefficients->get() : nullptr,
            config.bandpass.taps,
            config.bandpass.enabled);
        carotid_doppler::check_cuda(
            cudaEventRecord(bandpass_complete.get()),
            "record bandpass completion");
        iq_plan.transform(device_filtered.get(), device_iq.get());
        carotid_doppler::check_cuda(
            cudaEventRecord(iq_complete.get()), "record RF-to-IQ completion");
        carotid_doppler::check_cuda(
            cudaEventSynchronize(iq_complete.get()),
            "synchronize RF-to-IQ diagnostic");

        float tgc_ms = 0.0F;
        float bandpass_ms = 0.0F;
        float rf_to_iq_ms = 0.0F;
        carotid_doppler::check_cuda(
            cudaEventElapsedTime(
                &tgc_ms, start.get(), tgc_complete.get()),
            "measure TGC time");
        carotid_doppler::check_cuda(
            cudaEventElapsedTime(
                &bandpass_ms, tgc_complete.get(), bandpass_complete.get()),
            "measure bandpass time");
        carotid_doppler::check_cuda(
            cudaEventElapsedTime(
                &rf_to_iq_ms, bandpass_complete.get(), iq_complete.get()),
            "measure RF-to-IQ time");

        std::vector<float2> host_iq(shape.value_count());
        carotid_doppler::check_cuda(
            cudaMemcpy(
                host_iq.data(),
                device_iq.get(),
                device_iq.bytes(),
                cudaMemcpyDeviceToHost),
            "copy diagnostic IQ to host");
        float maximum_magnitude = 0.0F;
        for (std::size_t index = 0; index < host_iq.size(); ++index) {
            const float real = host_iq[index].x;
            const float imaginary = host_iq[index].y;
            if (!std::isfinite(real) || !std::isfinite(imaginary)) {
                throw std::runtime_error(
                    "RF-to-IQ produced non-finite value at complex index " +
                    std::to_string(index));
            }
            maximum_magnitude = std::max(
                maximum_magnitude, std::hypot(real, imaginary));
        }

        const std::filesystem::path iq_path =
            output_directory / "rf_iq_c64.bin";
        const auto write_begin = std::chrono::steady_clock::now();
        write_atomic_bytes(
            iq_path, host_iq.data(), host_iq.size() * sizeof(float2));
        const auto write_end = std::chrono::steady_clock::now();
        const std::string manifest = make_manifest(
            config,
            chunk,
            iq_path,
            host_iq.size() * sizeof(float2),
            iq_plan.workspace_bytes(),
            elapsed_ms(read_begin, read_end),
            tgc_ms,
            bandpass_ms,
            rf_to_iq_ms,
            elapsed_ms(write_begin, write_end),
            maximum_magnitude);
        write_atomic_text(
            output_directory / "iq_diagnostic_manifest.json", manifest);

        std::cout << "complete IQ diagnostic directory: "
                  << output_directory.string() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "carotid_iq_diagnostic: " << error.what() << '\n';
        return 2;
    }
}
