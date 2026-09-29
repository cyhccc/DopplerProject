#include "carotid_doppler/rf_bandpass.hpp"

#include <bit>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>

namespace carotid_doppler {
namespace {

std::size_t checked_multiply(
    std::size_t left,
    std::size_t right,
    const char* label) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        throw RfProcessingError(std::string(label) + " overflows size_t");
    }
    return left * right;
}

}  // namespace

std::size_t RfTensorShape::trace_count() const {
    validate();
    return checked_multiply(
        checked_multiply(frames, sequences, "RF trace count"),
        channels,
        "RF trace count");
}

std::size_t RfTensorShape::value_count() const {
    return checked_multiply(trace_count(), samples, "RF value count");
}

void RfTensorShape::validate() const {
    if (frames == 0 || sequences == 0 || channels == 0 || samples == 0) {
        throw RfProcessingError("RF tensor dimensions must all be positive");
    }
}

std::vector<float> load_bandpass_coefficients(
    const BandpassConfig& config) {
    if (config.dtype != "float32_le") {
        throw RfProcessingError(
            "bandpass coefficients must use dtype float32_le");
    }
    if (config.taps != kRequiredBandpassTaps) {
        throw RfProcessingError(
            "bandpass requires exactly 129 taps, got " +
            std::to_string(config.taps));
    }
    if (!config.enabled) {
        return {};
    }
    if constexpr (std::endian::native != std::endian::little) {
        throw RfProcessingError(
            "float32_le FIR coefficients require a little-endian host in v1");
    }
    if (!std::filesystem::exists(config.coefficients_path) ||
        !std::filesystem::is_regular_file(config.coefficients_path)) {
        throw RfProcessingError(
            "bandpass coefficient path is not a regular file: " +
            config.coefficients_path.string());
    }

    std::error_code error;
    const std::uintmax_t actual_bytes =
        std::filesystem::file_size(config.coefficients_path, error);
    if (error) {
        throw RfProcessingError(
            "failed to inspect bandpass coefficient file: " +
            error.message());
    }
    constexpr std::uintmax_t expected_bytes =
        kRequiredBandpassTaps * sizeof(float);
    if (actual_bytes != expected_bytes) {
        throw RfProcessingError(
            "bandpass coefficient file must contain exactly 129 little-endian "
            "float32 values (516 bytes), got " +
            std::to_string(actual_bytes) + " bytes");
    }

    std::vector<float> coefficients(kRequiredBandpassTaps);
    std::ifstream input(config.coefficients_path, std::ios::binary);
    if (!input) {
        throw RfProcessingError(
            "failed to open bandpass coefficient file: " +
            config.coefficients_path.string());
    }
    input.read(
        reinterpret_cast<char*>(coefficients.data()),
        static_cast<std::streamsize>(expected_bytes));
    if (input.gcount() != static_cast<std::streamsize>(expected_bytes)) {
        throw RfProcessingError("short read while loading FIR coefficients");
    }
    for (std::size_t index = 0; index < coefficients.size(); ++index) {
        if (!std::isfinite(coefficients[index])) {
            throw RfProcessingError(
                "bandpass coefficient " + std::to_string(index) +
                " is not finite");
        }
    }
    return coefficients;
}

RfTgcParameters make_rf_tgc_parameters(const PipelineConfig& config) {
    const double f0_mhz = config.acquisition.center_frequency_hz * 1.0e-6;
    const double alpha_db_cm = config.tgc.alpha_coeff_db_cm_mhz_power *
        std::pow(f0_mhz, config.tgc.frequency_power);
    const double alpha_np_m = alpha_db_cm / 14.686 * 100.0;
    const double scale = alpha_np_m *
        config.acquisition.sound_speed_m_per_s /
        config.acquisition.sample_rate_hz;
    const double offset = -alpha_np_m *
        config.acquisition.sound_speed_m_per_s *
        config.acquisition.transmit_cycle_count /
        config.acquisition.center_frequency_hz;
    if (!std::isfinite(scale) || !std::isfinite(offset) || scale < 0.0 ||
        scale > static_cast<double>(std::numeric_limits<float>::max()) ||
        std::abs(offset) >
            static_cast<double>(std::numeric_limits<float>::max())) {
        throw RfProcessingError("TGC parameters are not finite float32 values");
    }
    return RfTgcParameters{
        config.tgc.enabled,
        static_cast<float>(scale),
        static_cast<float>(offset)};
}

}  // namespace carotid_doppler
