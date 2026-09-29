#include "carotid_doppler/config.hpp"
#include "carotid_doppler/cuda_error.hpp"
#include "carotid_doppler/das_beamformer.hpp"
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
    bool dump_beamformed_iq{false};
    std::filesystem::path config_path;
    std::filesystem::path output_directory;
    std::size_t selected_frame_offset{0};
    std::size_t frame_count{1};
};

void print_usage(std::ostream& output) {
    output
        << "Usage:\n"
        << "  carotid_das_diagnostic --config <pipeline.json> "
           "--output-dir <new-directory> [--frame-offset <zero-based>] "
           "[--frame-count <positive>] [--dump-beamformed-iq]\n\n"
        << "The diagnostic consumes mapped ascending_x IQ, performs CUDA DAS, "
           "and always writes das_diagnostic_manifest.json. "
           "beamformed_iq_c64.bin is opt-in through either the CLI flag or "
           "output.dump_beamformed_iq=true. No dB data or image is written.\n";
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
        if (argument == "--dump-beamformed-iq") {
            options.dump_beamformed_iq = true;
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
            options.selected_frame_offset = parse_size(value, argument, true);
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
            "allocate DAS diagnostic device buffer");
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
            cudaEventCreate(&event_), "create DAS diagnostic CUDA event");
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

std::string tx_reference_name(carotid_doppler::TxReferenceMode mode) {
    return mode == carotid_doppler::TxReferenceMode::aperture_edge
        ? "aperture_edge"
        : "aligned_reference";
}

std::string make_manifest(
    const carotid_doppler::PipelineConfig& config,
    const carotid_doppler::MappedRfChunk& chunk,
    const carotid_doppler::DasParameters& das,
    bool cli_dump_requested,
    bool dump_enabled,
    std::size_t artifact_bytes,
    float maximum_magnitude,
    double read_ms,
    float tgc_ms,
    float bandpass_ms,
    float rf_to_iq_ms,
    float beamforming_ms,
    double write_ms) {
    const float dx =
        (das.x_max_m - das.x_min_m) / static_cast<float>(das.shape.nx - 1);
    const float dz =
        (das.z_max_m - das.z_min_m) / static_cast<float>(das.shape.nz - 1);
    std::ostringstream output;
    output << std::setprecision(10)
           << "{\n"
           << "  \"schema_version\": 1,\n"
           << "  \"kind\": \"das_diagnostic\",\n"
           << "  \"status\": \"complete\",\n"
           << "  \"config_path\": \""
           << json_escape(config.source_path.generic_string()) << "\",\n"
           << "  \"input\": {\n"
           << "    \"path\": \""
           << json_escape(config.input.path.generic_string()) << "\",\n"
           << "    \"source_frame_start\": " << chunk.source_frame_start
           << ",\n"
           << "    \"shape\": [" << chunk.frame_count << ", "
           << chunk.sequences << ", " << chunk.channels << ", "
           << chunk.samples << "],\n"
           << "    \"layout\": [\"frame\", \"sequence\", "
              "\"internal_channel\", \"sample\"],\n"
           << "    \"internal_channel_order\": \"ascending_x\"\n"
           << "  },\n"
           << "  \"geometry\": {\n"
           << "    \"sample_rate_hz\": " << das.sample_rate_hz << ",\n"
           << "    \"sound_speed_m_per_s\": " << das.sound_speed_m_per_s
           << ",\n"
           << "    \"pitch_m\": " << das.pitch_m << ",\n"
           << "    \"transmit_angles_deg\": [";
    for (std::size_t index = 0;
         index < config.acquisition.transmit_angles_deg.size(); ++index) {
        if (index != 0) output << ", ";
        output << config.acquisition.transmit_angles_deg[index];
    }
    output << "],\n"
           << "    \"tx_delay_offset_samples\": "
           << das.tx_delay_offset_samples << ",\n"
           << "    \"tx_reference_mode\": \""
           << tx_reference_name(das.tx_reference_mode) << "\",\n"
           << "    \"f_number\": " << das.f_number << ",\n"
           << "    \"minimum_aperture_m\": " << das.minimum_aperture_m
           << "\n"
           << "  },\n"
           << "  \"image_grid\": {\n"
           << "    \"nx\": " << das.shape.nx << ",\n"
           << "    \"nz\": " << das.shape.nz << ",\n"
           << "    \"x_min_m\": " << das.x_min_m << ",\n"
           << "    \"x_max_m\": " << das.x_max_m << ",\n"
           << "    \"z_min_m\": " << das.z_min_m << ",\n"
           << "    \"z_max_m\": " << das.z_max_m << ",\n"
           << "    \"dx_m\": " << dx << ",\n"
           << "    \"dz_m\": " << dz << ",\n"
           << "    \"layout\": [\"z\", \"x\"]\n"
           << "  },\n"
           << "  \"processing\": {\n"
           << "    \"tgc_enabled\": "
           << (config.tgc.enabled ? "true" : "false") << ",\n"
           << "    \"preprocessing_order\": \"int16_map,tgc,centered_bandpass,hilbert\",\n"
           << "    \"bandpass_enabled\": "
           << (config.bandpass.enabled ? "true" : "false") << ",\n"
           << "    \"iq_phase_convention\": "
              "\"real=filtered_rf, imag=Hilbert(filtered_rf)\",\n"
           << "    \"fractional_delay\": \"complex_linear_interpolation\",\n"
           << "    \"receive_aperture\": \"dynamic_"
           << config.receive.apodization << "\",\n"
           << "    \"sequence_combination\": \"coherent_sum\",\n"
           << "    \"direction_correction\": \"none\"\n"
           << "  },\n"
           << "  \"artifact\": ";
    if (dump_enabled) {
        output << "{\n"
               << "    \"path\": \"beamformed_iq_c64.bin\",\n"
               << "    \"scalar_dtype\": \"float32_le\",\n"
               << "    \"complex_encoding\": "
                  "\"interleaved_real_imag\",\n"
               << "    \"shape\": [" << das.shape.frames << ", "
               << das.shape.nz << ", " << das.shape.nx << "],\n"
               << "    \"layout\": [\"frame\", \"z\", \"x\"],\n"
               << "    \"contiguous_axis\": \"x\",\n"
               << "    \"complex_index\": "
                  "\"x+z*nx+frame*nx*nz\",\n"
               << "    \"size_bytes\": " << artifact_bytes << ",\n"
               << "    \"maximum_magnitude\": " << maximum_magnitude
               << "\n"
               << "  },\n";
    } else {
        output << "null,\n";
    }
    output << "  \"diagnostic_options\": {\n"
           << "    \"cli_dump_beamformed_iq\": "
           << (cli_dump_requested ? "true" : "false") << ",\n"
           << "    \"config_dump_beamformed_iq\": "
           << (config.output.dump_beamformed_iq ? "true" : "false")
           << ",\n"
           << "    \"dump_beamformed_iq\": "
           << (dump_enabled ? "true" : "false") << "\n"
           << "  },\n"
           << "  \"timings_ms\": {\n"
           << "    \"read_and_map\": " << read_ms << ",\n"
           << "    \"tgc_or_map\": " << tgc_ms << ",\n"
           << "    \"bandpass_or_copy\": " << bandpass_ms << ",\n"
           << "    \"rf_to_iq\": " << rf_to_iq_ms << ",\n"
           << "    \"beamforming\": " << beamforming_ms << ",\n"
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
                "DAS diagnostic binary contract requires a little-endian host");
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
        const bool dump_beamformed_iq =
            options.dump_beamformed_iq || config.output.dump_beamformed_iq;

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

        const carotid_doppler::RfTensorShape rf_shape{
            chunk.frame_count,
            chunk.sequences,
            chunk.channels,
            chunk.samples};
        if (chunk.values.size() != rf_shape.value_count()) {
            throw std::runtime_error(
                "mapped RF chunk does not match declared internal shape");
        }
        const auto das_parameters =
            carotid_doppler::make_das_parameters(config, chunk.frame_count);

        DeviceBuffer<std::int16_t> device_rf(rf_shape.value_count());
        DeviceBuffer<float> device_tgc(rf_shape.value_count());
        DeviceBuffer<float> device_filtered(rf_shape.value_count());
        DeviceBuffer<float2> device_iq(rf_shape.value_count());
        DeviceBuffer<float2> device_beamformed(
            das_parameters.shape.output_value_count());
        carotid_doppler::check_cuda(
            cudaMemcpy(
                device_rf.get(),
                chunk.values.data(),
                device_rf.bytes(),
                cudaMemcpyHostToDevice),
            "copy mapped RF to device for DAS diagnostic");

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
                "copy FIR coefficients to device for DAS diagnostic");
        }

        carotid_doppler::RfToIqPlan iq_plan(rf_shape);
        carotid_doppler::DasBeamformer beamformer(das_parameters);
        CudaEvent start;
        CudaEvent tgc_complete;
        CudaEvent bandpass_complete;
        CudaEvent iq_complete;
        CudaEvent beamforming_complete;
        carotid_doppler::check_cuda(
            cudaEventRecord(start.get()), "record DAS diagnostic start");
        carotid_doppler::map_rf_and_apply_tgc(
            device_rf.get(),
            device_tgc.get(),
            rf_shape,
            carotid_doppler::make_rf_tgc_parameters(config));
        carotid_doppler::check_cuda(
            cudaEventRecord(tgc_complete.get()),
            "record DAS diagnostic TGC completion");
        carotid_doppler::apply_centered_rf_bandpass(
            device_tgc.get(),
            device_filtered.get(),
            rf_shape,
            device_coefficients ? device_coefficients->get() : nullptr,
            config.bandpass.taps,
            config.bandpass.enabled);
        carotid_doppler::check_cuda(
            cudaEventRecord(bandpass_complete.get()),
            "record DAS diagnostic bandpass completion");
        iq_plan.transform(device_filtered.get(), device_iq.get());
        carotid_doppler::check_cuda(
            cudaEventRecord(iq_complete.get()),
            "record DAS diagnostic IQ completion");
        beamformer.beamform(device_iq.get(), device_beamformed.get());
        carotid_doppler::check_cuda(
            cudaEventRecord(beamforming_complete.get()),
            "record DAS diagnostic beamforming completion");
        carotid_doppler::check_cuda(
            cudaEventSynchronize(beamforming_complete.get()),
            "synchronize DAS diagnostic");

        float tgc_ms = 0.0F;
        float bandpass_ms = 0.0F;
        float rf_to_iq_ms = 0.0F;
        float beamforming_ms = 0.0F;
        carotid_doppler::check_cuda(
            cudaEventElapsedTime(
                &tgc_ms, start.get(), tgc_complete.get()),
            "measure DAS diagnostic TGC time");
        carotid_doppler::check_cuda(
            cudaEventElapsedTime(
                &bandpass_ms, tgc_complete.get(), bandpass_complete.get()),
            "measure DAS diagnostic bandpass time");
        carotid_doppler::check_cuda(
            cudaEventElapsedTime(
                &rf_to_iq_ms, bandpass_complete.get(), iq_complete.get()),
            "measure DAS diagnostic IQ time");
        carotid_doppler::check_cuda(
            cudaEventElapsedTime(
                &beamforming_ms,
                iq_complete.get(),
                beamforming_complete.get()),
            "measure DAS diagnostic beamforming time");

        std::size_t artifact_bytes = 0;
        float maximum_magnitude = 0.0F;
        double write_ms = 0.0;
        if (dump_beamformed_iq) {
            std::vector<float2> host_beamformed(
                das_parameters.shape.output_value_count());
            carotid_doppler::check_cuda(
                cudaMemcpy(
                    host_beamformed.data(),
                    device_beamformed.get(),
                    device_beamformed.bytes(),
                    cudaMemcpyDeviceToHost),
                "copy beamformed diagnostic IQ to host");
            for (std::size_t index = 0; index < host_beamformed.size(); ++index) {
                const float real = host_beamformed[index].x;
                const float imaginary = host_beamformed[index].y;
                if (!std::isfinite(real) || !std::isfinite(imaginary)) {
                    throw std::runtime_error(
                        "DAS produced non-finite value at complex index " +
                        std::to_string(index));
                }
                maximum_magnitude = std::max(
                    maximum_magnitude, std::hypot(real, imaginary));
            }
            artifact_bytes = host_beamformed.size() * sizeof(float2);
            const auto write_begin = std::chrono::steady_clock::now();
            write_atomic_bytes(
                output_directory / "beamformed_iq_c64.bin",
                host_beamformed.data(),
                artifact_bytes);
            const auto write_end = std::chrono::steady_clock::now();
            write_ms = elapsed_ms(write_begin, write_end);
        }

        const std::string manifest = make_manifest(
            config,
            chunk,
            das_parameters,
            options.dump_beamformed_iq,
            dump_beamformed_iq,
            artifact_bytes,
            maximum_magnitude,
            elapsed_ms(read_begin, read_end),
            tgc_ms,
            bandpass_ms,
            rf_to_iq_ms,
            beamforming_ms,
            write_ms);
        write_atomic_text(
            output_directory / "das_diagnostic_manifest.json", manifest);

        std::cout << "complete DAS diagnostic directory: "
                  << output_directory.string() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "carotid_das_diagnostic: " << error.what() << '\n';
        return 2;
    }
}
