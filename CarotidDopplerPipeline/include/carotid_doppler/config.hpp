#pragma once

#include <cstddef>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace carotid_doppler {

class ConfigError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

enum class ChannelOrder {
    ascending_x,
    descending_x,
};

std::string_view channel_order_name(ChannelOrder order) noexcept;

struct RfInputConfig {
    std::filesystem::path path;
    std::string dtype;
    std::vector<std::string> layout;
    std::size_t sequences{};
    std::size_t channels{};
    std::size_t samples{};
    std::size_t frame_start{};
    std::size_t frame_count{};
    ChannelOrder channel_order{ChannelOrder::ascending_x};
};

struct AcquisitionConfig {
    double sample_rate_hz{};
    double center_frequency_hz{};
    double sound_speed_m_per_s{};
    double pitch_m{};
    std::vector<double> transmit_angles_deg;
    std::size_t tx_delay_offset_samples{};
    double transmit_cycle_count{};
    std::string tx_reference_mode;
};

struct ImageGridConfig {
    std::size_t nx{};
    std::size_t nz{};
    double x_min_m{};
    double x_max_m{};
    double z_min_m{};
    double z_max_m{};
};

struct ReceiveConfig {
    double f_number{};
    double minimum_aperture_m{};
    std::string apodization;
};

struct TgcConfig {
    bool enabled{};
    double alpha_coeff_db_cm_mhz_power{};
    double frequency_power{};
};

struct BandpassConfig {
    bool enabled{};
    std::filesystem::path coefficients_path;
    std::string dtype;
    std::size_t taps{};
};

struct SvdConfig {
    std::string mode;
    std::size_t keep_start{};
    std::size_t keep_end{};
    std::string indexing;
};

struct OutputConfig {
    std::filesystem::path run_root;
    bool dump_beamformed_iq{};
    bool dump_filtered_iq{};
};

struct PipelineConfig {
    std::filesystem::path source_path;
    int schema_version{};
    RfInputConfig input;
    AcquisitionConfig acquisition;
    ImageGridConfig image_grid;
    ReceiveConfig receive;
    TgcConfig tgc;
    BandpassConfig bandpass;
    SvdConfig svd;
    OutputConfig output;
};

PipelineConfig load_pipeline_config(const std::filesystem::path& config_path);

}  // namespace carotid_doppler
