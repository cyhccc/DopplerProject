#include "carotid_doppler/config.hpp"

#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <system_error>
#include <variant>

namespace carotid_doppler {
namespace {

struct JsonValue {
    using Array = std::vector<JsonValue>;
    using Object = std::map<std::string, JsonValue, std::less<>>;
    using Storage =
        std::variant<std::nullptr_t, bool, double, std::string, Array, Object>;

    Storage storage;
};

class JsonParser {
public:
    explicit JsonParser(std::string_view source) : source_(source) {}

    JsonValue parse() {
        skip_whitespace();
        JsonValue value = parse_value();
        skip_whitespace();
        if (position_ != source_.size()) {
            fail("unexpected trailing content");
        }
        return value;
    }

private:
    JsonValue parse_value() {
        if (position_ >= source_.size()) {
            fail("expected a JSON value");
        }
        switch (source_[position_]) {
        case '{': return JsonValue{parse_object()};
        case '[': return JsonValue{parse_array()};
        case '"': return JsonValue{parse_string()};
        case 't': consume_literal("true"); return JsonValue{true};
        case 'f': consume_literal("false"); return JsonValue{false};
        case 'n': consume_literal("null"); return JsonValue{nullptr};
        default:
            if (source_[position_] == '-' || is_digit(source_[position_])) {
                return JsonValue{parse_number()};
            }
            fail("expected object, array, string, number, boolean, or null");
        }
    }

    JsonValue::Object parse_object() {
        expect('{');
        skip_whitespace();
        JsonValue::Object object;
        if (consume_if('}')) {
            return object;
        }
        while (true) {
            if (position_ >= source_.size() || source_[position_] != '"') {
                fail("expected a quoted object key");
            }
            std::string key = parse_string();
            skip_whitespace();
            expect(':');
            skip_whitespace();
            JsonValue value = parse_value();
            if (!object.emplace(key, std::move(value)).second) {
                fail("duplicate object key '" + key + "'");
            }
            skip_whitespace();
            if (consume_if('}')) {
                return object;
            }
            expect(',');
            skip_whitespace();
        }
    }

    JsonValue::Array parse_array() {
        expect('[');
        skip_whitespace();
        JsonValue::Array array;
        if (consume_if(']')) {
            return array;
        }
        while (true) {
            array.push_back(parse_value());
            skip_whitespace();
            if (consume_if(']')) {
                return array;
            }
            expect(',');
            skip_whitespace();
        }
    }

    std::string parse_string() {
        expect('"');
        std::string value;
        while (position_ < source_.size()) {
            const unsigned char character =
                static_cast<unsigned char>(source_[position_++]);
            if (character == '"') {
                return value;
            }
            if (character < 0x20U) {
                fail("unescaped control character in string");
            }
            if (character != '\\') {
                value.push_back(static_cast<char>(character));
                continue;
            }
            if (position_ >= source_.size()) {
                fail("unterminated string escape");
            }
            const char escape = source_[position_++];
            switch (escape) {
            case '"': value.push_back('"'); break;
            case '\\': value.push_back('\\'); break;
            case '/': value.push_back('/'); break;
            case 'b': value.push_back('\b'); break;
            case 'f': value.push_back('\f'); break;
            case 'n': value.push_back('\n'); break;
            case 'r': value.push_back('\r'); break;
            case 't': value.push_back('\t'); break;
            case 'u': append_unicode_escape(value); break;
            default: fail("invalid string escape");
            }
        }
        fail("unterminated string");
    }

    void append_unicode_escape(std::string& output) {
        std::uint32_t code_point = parse_hex_quad();
        if (code_point >= 0xD800U && code_point <= 0xDBFFU) {
            if (position_ + 2 > source_.size() ||
                source_.substr(position_, 2) != "\\u") {
                fail("high surrogate must be followed by a low surrogate");
            }
            position_ += 2;
            const std::uint32_t low = parse_hex_quad();
            if (low < 0xDC00U || low > 0xDFFFU) {
                fail("invalid low surrogate");
            }
            code_point =
                0x10000U + ((code_point - 0xD800U) << 10U) + (low - 0xDC00U);
        } else if (code_point >= 0xDC00U && code_point <= 0xDFFFU) {
            fail("unexpected low surrogate");
        }
        append_utf8(output, code_point);
    }

    std::uint32_t parse_hex_quad() {
        if (position_ + 4 > source_.size()) {
            fail("incomplete unicode escape");
        }
        std::uint32_t value = 0;
        for (int index = 0; index < 4; ++index) {
            const char character = source_[position_++];
            value <<= 4U;
            if (character >= '0' && character <= '9') {
                value += static_cast<std::uint32_t>(character - '0');
            } else if (character >= 'a' && character <= 'f') {
                value += static_cast<std::uint32_t>(character - 'a' + 10);
            } else if (character >= 'A' && character <= 'F') {
                value += static_cast<std::uint32_t>(character - 'A' + 10);
            } else {
                fail("invalid hexadecimal digit in unicode escape");
            }
        }
        return value;
    }

    static void append_utf8(std::string& output, std::uint32_t code_point) {
        if (code_point <= 0x7FU) {
            output.push_back(static_cast<char>(code_point));
        } else if (code_point <= 0x7FFU) {
            output.push_back(static_cast<char>(0xC0U | (code_point >> 6U)));
            output.push_back(static_cast<char>(0x80U | (code_point & 0x3FU)));
        } else if (code_point <= 0xFFFFU) {
            output.push_back(static_cast<char>(0xE0U | (code_point >> 12U)));
            output.push_back(
                static_cast<char>(0x80U | ((code_point >> 6U) & 0x3FU)));
            output.push_back(static_cast<char>(0x80U | (code_point & 0x3FU)));
        } else {
            output.push_back(static_cast<char>(0xF0U | (code_point >> 18U)));
            output.push_back(
                static_cast<char>(0x80U | ((code_point >> 12U) & 0x3FU)));
            output.push_back(
                static_cast<char>(0x80U | ((code_point >> 6U) & 0x3FU)));
            output.push_back(static_cast<char>(0x80U | (code_point & 0x3FU)));
        }
    }

    double parse_number() {
        const std::size_t begin = position_;
        consume_if('-');
        if (consume_if('0')) {
            if (position_ < source_.size() && is_digit(source_[position_])) {
                fail("leading zero is not permitted in JSON number");
            }
        } else {
            require_digits();
        }
        if (consume_if('.')) {
            require_digits();
        }
        if (consume_if('e') || consume_if('E')) {
            const bool sign_consumed = consume_if('+') || consume_if('-');
            (void)sign_consumed;
            require_digits();
        }
        const std::string_view token =
            source_.substr(begin, position_ - begin);
        double value = 0.0;
        const auto result =
            std::from_chars(token.data(), token.data() + token.size(), value);
        if (result.ec != std::errc{} ||
            result.ptr != token.data() + token.size() || !std::isfinite(value)) {
            fail("invalid or non-finite JSON number");
        }
        return value;
    }

    void require_digits() {
        const std::size_t begin = position_;
        while (position_ < source_.size() && is_digit(source_[position_])) {
            ++position_;
        }
        if (position_ == begin) {
            fail("expected digit in JSON number");
        }
    }

    void consume_literal(std::string_view literal) {
        if (source_.substr(position_, literal.size()) != literal) {
            fail("invalid JSON literal");
        }
        position_ += literal.size();
    }

    bool consume_if(char expected) {
        if (position_ < source_.size() && source_[position_] == expected) {
            ++position_;
            return true;
        }
        return false;
    }

    void expect(char expected) {
        if (!consume_if(expected)) {
            fail(std::string("expected '") + expected + "'");
        }
    }

    void skip_whitespace() {
        while (position_ < source_.size()) {
            const char character = source_[position_];
            if (character != ' ' && character != '\t' && character != '\r' &&
                character != '\n') {
                break;
            }
            ++position_;
        }
    }

    [[noreturn]] void fail(const std::string& message) const {
        throw ConfigError(
            "JSON parse error at byte " + std::to_string(position_) + ": " +
            message);
    }

    static bool is_digit(char character) noexcept {
        return character >= '0' && character <= '9';
    }

    std::string_view source_;
    std::size_t position_{};
};

std::string type_name(const JsonValue& value) {
    if (std::holds_alternative<std::nullptr_t>(value.storage)) return "null";
    if (std::holds_alternative<bool>(value.storage)) return "boolean";
    if (std::holds_alternative<double>(value.storage)) return "number";
    if (std::holds_alternative<std::string>(value.storage)) return "string";
    if (std::holds_alternative<JsonValue::Array>(value.storage)) return "array";
    return "object";
}

[[noreturn]] void contract_error(
    std::string_view path,
    std::string_view actual,
    std::string_view expected) {
    throw ConfigError(
        std::string(path) + ": actual " + std::string(actual) + ", expected " +
        std::string(expected));
}

const JsonValue::Object& require_object(
    const JsonValue& value,
    std::string_view path) {
    if (const auto* object = std::get_if<JsonValue::Object>(&value.storage)) {
        return *object;
    }
    contract_error(path, type_name(value), "object");
}

const JsonValue::Array& require_array(
    const JsonValue& value,
    std::string_view path) {
    if (const auto* array = std::get_if<JsonValue::Array>(&value.storage)) {
        return *array;
    }
    contract_error(path, type_name(value), "array");
}

const JsonValue& require_member(
    const JsonValue::Object& object,
    std::string_view key,
    std::string_view object_path) {
    const auto iterator = object.find(key);
    if (iterator == object.end()) {
        throw ConfigError(
            std::string(object_path) + "." + std::string(key) +
            ": actual missing, expected required field");
    }
    return iterator->second;
}

template <std::size_t Size>
void reject_unknown_fields(
    const JsonValue::Object& object,
    const std::array<std::string_view, Size>& allowed,
    std::string_view object_path) {
    for (const auto& [key, value] : object) {
        (void)value;
        bool known = false;
        for (const std::string_view candidate : allowed) {
            if (candidate == key) {
                known = true;
                break;
            }
        }
        if (!known) {
            std::ostringstream expected;
            expected << "only fields [";
            for (std::size_t index = 0; index < allowed.size(); ++index) {
                if (index != 0) expected << ", ";
                expected << allowed[index];
            }
            expected << "]";
            throw ConfigError(
                std::string(object_path) + "." + key +
                ": unknown field; actual field present, expected " +
                expected.str());
        }
    }
}

std::string require_string(const JsonValue& value, std::string_view path) {
    if (const auto* text = std::get_if<std::string>(&value.storage)) {
        if (text->empty()) {
            contract_error(path, "empty string", "non-empty string");
        }
        return *text;
    }
    contract_error(path, type_name(value), "string");
}

bool require_bool(const JsonValue& value, std::string_view path) {
    if (const auto* boolean = std::get_if<bool>(&value.storage)) {
        return *boolean;
    }
    contract_error(path, type_name(value), "boolean");
}

double require_number(const JsonValue& value, std::string_view path) {
    if (const auto* number = std::get_if<double>(&value.storage)) {
        if (std::isfinite(*number)) {
            return *number;
        }
    }
    contract_error(path, type_name(value), "finite number");
}

double require_positive_number(const JsonValue& value, std::string_view path) {
    const double number = require_number(value, path);
    if (!(number > 0.0)) {
        contract_error(
            path, std::to_string(number), "number greater than zero");
    }
    return number;
}

std::size_t require_size(
    const JsonValue& value,
    std::string_view path,
    bool allow_zero = false) {
    const double number = require_number(value, path);
    const double minimum = allow_zero ? 0.0 : 1.0;
    if (number < minimum || std::floor(number) != number ||
        number > static_cast<double>(
                     std::numeric_limits<std::size_t>::max())) {
        contract_error(
            path,
            std::to_string(number),
            allow_zero ? "non-negative integer" : "positive integer");
    }
    return static_cast<std::size_t>(number);
}

std::filesystem::path resolve_path(
    const std::filesystem::path& config_path,
    const std::string& text) {
    std::filesystem::path path(text);
    if (path.is_relative()) {
        path = config_path.parent_path() / path;
    }
    return std::filesystem::absolute(path).lexically_normal();
}

std::vector<std::string> parse_layout(const JsonValue& value) {
    const auto& array = require_array(value, "$.input.layout");
    const std::array<std::string_view, 4> expected{
        "frame", "sequence", "sample", "source_channel"};
    if (array.size() != expected.size()) {
        contract_error(
            "$.input.layout",
            std::to_string(array.size()) + " items",
            "[frame, sequence, sample, source_channel]");
    }
    std::vector<std::string> layout;
    layout.reserve(array.size());
    for (std::size_t index = 0; index < array.size(); ++index) {
        const std::string path =
            "$.input.layout[" + std::to_string(index) + "]";
        const std::string axis = require_string(array[index], path);
        if (axis != expected[index]) {
            contract_error(
                path,
                "'" + axis + "'",
                "'" + std::string(expected[index]) + "'");
        }
        layout.push_back(axis);
    }
    return layout;
}

RfInputConfig parse_input(
    const JsonValue& value,
    const std::filesystem::path& config_path) {
    const auto& object = require_object(value, "$.input");
    constexpr std::array allowed{
        std::string_view("path"),
        std::string_view("dtype"),
        std::string_view("layout"),
        std::string_view("sequences"),
        std::string_view("channels"),
        std::string_view("samples"),
        std::string_view("frame_start"),
        std::string_view("frame_count"),
        std::string_view("channel_order")};
    reject_unknown_fields(object, allowed, "$.input");

    RfInputConfig input;
    input.path = resolve_path(
        config_path,
        require_string(
            require_member(object, "path", "$.input"), "$.input.path"));
    input.dtype = require_string(
        require_member(object, "dtype", "$.input"), "$.input.dtype");
    if (input.dtype != "int16_le") {
        contract_error(
            "$.input.dtype", "'" + input.dtype + "'", "'int16_le'");
    }
    input.layout = parse_layout(require_member(object, "layout", "$.input"));
    input.sequences = require_size(
        require_member(object, "sequences", "$.input"), "$.input.sequences");
    input.channels = require_size(
        require_member(object, "channels", "$.input"), "$.input.channels");
    input.samples = require_size(
        require_member(object, "samples", "$.input"), "$.input.samples");
    input.frame_start = require_size(
        require_member(object, "frame_start", "$.input"),
        "$.input.frame_start",
        true);
    input.frame_count = require_size(
        require_member(object, "frame_count", "$.input"),
        "$.input.frame_count");
    const std::string order = require_string(
        require_member(object, "channel_order", "$.input"),
        "$.input.channel_order");
    if (order == "ascending_x") {
        input.channel_order = ChannelOrder::ascending_x;
    } else if (order == "descending_x") {
        input.channel_order = ChannelOrder::descending_x;
    } else {
        contract_error(
            "$.input.channel_order",
            "'" + order + "'",
            "'ascending_x' or 'descending_x'");
    }
    return input;
}

AcquisitionConfig parse_acquisition(
    const JsonValue& value,
    std::size_t sequences) {
    const auto& object = require_object(value, "$.acquisition");
    constexpr std::array allowed{
        std::string_view("sample_rate_hz"),
        std::string_view("center_frequency_hz"),
        std::string_view("sound_speed_m_per_s"),
        std::string_view("pitch_m"),
        std::string_view("transmit_angles_deg"),
        std::string_view("tx_delay_offset_samples"),
        std::string_view("transmit_cycle_count"),
        std::string_view("tx_reference_mode")};
    reject_unknown_fields(object, allowed, "$.acquisition");

    AcquisitionConfig acquisition;
    acquisition.sample_rate_hz = require_positive_number(
        require_member(object, "sample_rate_hz", "$.acquisition"),
        "$.acquisition.sample_rate_hz");
    acquisition.center_frequency_hz = require_positive_number(
        require_member(object, "center_frequency_hz", "$.acquisition"),
        "$.acquisition.center_frequency_hz");
    acquisition.sound_speed_m_per_s = require_positive_number(
        require_member(object, "sound_speed_m_per_s", "$.acquisition"),
        "$.acquisition.sound_speed_m_per_s");
    acquisition.pitch_m = require_positive_number(
        require_member(object, "pitch_m", "$.acquisition"),
        "$.acquisition.pitch_m");
    const auto& angles = require_array(
        require_member(object, "transmit_angles_deg", "$.acquisition"),
        "$.acquisition.transmit_angles_deg");
    if (angles.size() != sequences) {
        contract_error(
            "$.acquisition.transmit_angles_deg",
            std::to_string(angles.size()) + " items",
            std::to_string(sequences) +
                " items to match $.input.sequences");
    }
    acquisition.transmit_angles_deg.reserve(angles.size());
    for (std::size_t index = 0; index < angles.size(); ++index) {
        acquisition.transmit_angles_deg.push_back(require_number(
            angles[index],
            "$.acquisition.transmit_angles_deg[" +
                std::to_string(index) + "]"));
    }
    acquisition.tx_delay_offset_samples = require_size(
        require_member(object, "tx_delay_offset_samples", "$.acquisition"),
        "$.acquisition.tx_delay_offset_samples",
        true);
    acquisition.transmit_cycle_count = require_positive_number(
        require_member(object, "transmit_cycle_count", "$.acquisition"),
        "$.acquisition.transmit_cycle_count");
    acquisition.tx_reference_mode = require_string(
        require_member(object, "tx_reference_mode", "$.acquisition"),
        "$.acquisition.tx_reference_mode");
    if (acquisition.tx_reference_mode != "aperture_edge" &&
        acquisition.tx_reference_mode != "aligned_reference") {
        contract_error(
            "$.acquisition.tx_reference_mode",
            "'" + acquisition.tx_reference_mode + "'",
            "'aperture_edge' or 'aligned_reference'");
    }
    return acquisition;
}

ImageGridConfig parse_image_grid(const JsonValue& value) {
    const auto& object = require_object(value, "$.image_grid");
    constexpr std::array allowed{
        std::string_view("nx"),
        std::string_view("nz"),
        std::string_view("x_min_m"),
        std::string_view("x_max_m"),
        std::string_view("z_min_m"),
        std::string_view("z_max_m")};
    reject_unknown_fields(object, allowed, "$.image_grid");
    ImageGridConfig grid;
    grid.nx = require_size(
        require_member(object, "nx", "$.image_grid"), "$.image_grid.nx");
    grid.nz = require_size(
        require_member(object, "nz", "$.image_grid"), "$.image_grid.nz");
    if (grid.nx < 2 || grid.nz < 2) {
        contract_error(
            "$.image_grid",
            "nx or nz below 2",
            "nx >= 2 and nz >= 2");
    }
    grid.x_min_m = require_number(
        require_member(object, "x_min_m", "$.image_grid"),
        "$.image_grid.x_min_m");
    grid.x_max_m = require_number(
        require_member(object, "x_max_m", "$.image_grid"),
        "$.image_grid.x_max_m");
    grid.z_min_m = require_number(
        require_member(object, "z_min_m", "$.image_grid"),
        "$.image_grid.z_min_m");
    grid.z_max_m = require_number(
        require_member(object, "z_max_m", "$.image_grid"),
        "$.image_grid.z_max_m");
    if (!(grid.x_min_m < grid.x_max_m)) {
        contract_error(
            "$.image_grid",
            "x_min_m >= x_max_m",
            "x_min_m < x_max_m");
    }
    if (!(grid.z_min_m < grid.z_max_m)) {
        contract_error(
            "$.image_grid",
            "z_min_m >= z_max_m",
            "z_min_m < z_max_m");
    }
    return grid;
}

ReceiveConfig parse_receive(const JsonValue& value) {
    const auto& object = require_object(value, "$.receive");
    constexpr std::array allowed{
        std::string_view("f_number"),
        std::string_view("minimum_aperture_m"),
        std::string_view("apodization")};
    reject_unknown_fields(object, allowed, "$.receive");
    ReceiveConfig receive{
        require_positive_number(
            require_member(object, "f_number", "$.receive"),
            "$.receive.f_number"),
        require_positive_number(
            require_member(object, "minimum_aperture_m", "$.receive"),
            "$.receive.minimum_aperture_m"),
        require_string(
            require_member(object, "apodization", "$.receive"),
            "$.receive.apodization")};
    if (receive.apodization != "rectangular" &&
        receive.apodization != "tukey25") {
        contract_error(
            "$.receive.apodization",
            "'" + receive.apodization + "'",
            "'rectangular' or 'tukey25'");
    }
    return receive;
}

TgcConfig parse_tgc(const JsonValue& value) {
    const auto& object = require_object(value, "$.tgc");
    constexpr std::array allowed{
        std::string_view("enabled"),
        std::string_view("alpha_coeff_db_cm_mhz_power"),
        std::string_view("frequency_power")};
    reject_unknown_fields(object, allowed, "$.tgc");
    return TgcConfig{
        require_bool(
            require_member(object, "enabled", "$.tgc"),
            "$.tgc.enabled"),
        require_positive_number(
            require_member(
                object, "alpha_coeff_db_cm_mhz_power", "$.tgc"),
            "$.tgc.alpha_coeff_db_cm_mhz_power"),
        require_positive_number(
            require_member(object, "frequency_power", "$.tgc"),
            "$.tgc.frequency_power")};
}

BandpassConfig parse_bandpass(
    const JsonValue& value,
    const std::filesystem::path& config_path) {
    const auto& object = require_object(value, "$.bandpass");
    constexpr std::array allowed{
        std::string_view("enabled"),
        std::string_view("coefficients_path"),
        std::string_view("dtype"),
        std::string_view("taps")};
    reject_unknown_fields(object, allowed, "$.bandpass");
    BandpassConfig bandpass;
    bandpass.enabled = require_bool(
        require_member(object, "enabled", "$.bandpass"),
        "$.bandpass.enabled");
    bandpass.coefficients_path = resolve_path(
        config_path,
        require_string(
            require_member(object, "coefficients_path", "$.bandpass"),
            "$.bandpass.coefficients_path"));
    bandpass.dtype = require_string(
        require_member(object, "dtype", "$.bandpass"),
        "$.bandpass.dtype");
    if (bandpass.dtype != "float32_le") {
        contract_error(
            "$.bandpass.dtype",
            "'" + bandpass.dtype + "'",
            "'float32_le'");
    }
    bandpass.taps = require_size(
        require_member(object, "taps", "$.bandpass"),
        "$.bandpass.taps");
    return bandpass;
}

SvdConfig parse_svd(const JsonValue& value, std::size_t frame_count) {
    const auto& object = require_object(value, "$.svd");
    constexpr std::array allowed{
        std::string_view("mode"),
        std::string_view("keep_start"),
        std::string_view("keep_end"),
        std::string_view("indexing")};
    reject_unknown_fields(object, allowed, "$.svd");
    SvdConfig svd;
    svd.mode =
        require_string(require_member(object, "mode", "$.svd"), "$.svd.mode");
    if (svd.mode != "global_temporal") {
        contract_error(
            "$.svd.mode", "'" + svd.mode + "'", "'global_temporal'");
    }
    svd.keep_start = require_size(
        require_member(object, "keep_start", "$.svd"),
        "$.svd.keep_start");
    svd.keep_end = require_size(
        require_member(object, "keep_end", "$.svd"),
        "$.svd.keep_end");
    svd.indexing = require_string(
        require_member(object, "indexing", "$.svd"),
        "$.svd.indexing");
    if (svd.indexing != "one_based_inclusive") {
        contract_error(
            "$.svd.indexing",
            "'" + svd.indexing + "'",
            "'one_based_inclusive'");
    }
    if (svd.keep_start > svd.keep_end || svd.keep_end > frame_count) {
        contract_error(
            "$.svd",
            "keep range [" + std::to_string(svd.keep_start) + "," +
                std::to_string(svd.keep_end) + "]",
            "1 <= keep_start <= keep_end <= $.input.frame_count");
    }
    return svd;
}

OutputConfig parse_output(
    const JsonValue& value,
    const std::filesystem::path& config_path) {
    const auto& object = require_object(value, "$.output");
    constexpr std::array allowed{
        std::string_view("run_root"),
        std::string_view("dump_beamformed_iq"),
        std::string_view("dump_filtered_iq")};
    reject_unknown_fields(object, allowed, "$.output");
    return OutputConfig{
        resolve_path(
            config_path,
            require_string(
                require_member(object, "run_root", "$.output"),
                "$.output.run_root")),
        require_bool(
            require_member(object, "dump_beamformed_iq", "$.output"),
            "$.output.dump_beamformed_iq"),
        require_bool(
            require_member(object, "dump_filtered_iq", "$.output"),
            "$.output.dump_filtered_iq")};
}

std::string read_utf8_text(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw ConfigError(
            "config path: actual unreadable file '" + path.string() +
            "', expected existing readable JSON file");
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    if (!input.good() && !input.eof()) {
        throw ConfigError("failed while reading config file: " + path.string());
    }
    std::string content = buffer.str();
    constexpr std::array<unsigned char, 3> utf8_bom{
        0xEFU, 0xBBU, 0xBFU};
    if (content.size() >= utf8_bom.size() &&
        static_cast<unsigned char>(content[0]) == utf8_bom[0] &&
        static_cast<unsigned char>(content[1]) == utf8_bom[1] &&
        static_cast<unsigned char>(content[2]) == utf8_bom[2]) {
        content.erase(0, utf8_bom.size());
    }
    return content;
}

}  // namespace

std::string_view channel_order_name(ChannelOrder order) noexcept {
    return order == ChannelOrder::ascending_x ? "ascending_x" : "descending_x";
}

PipelineConfig load_pipeline_config(
    const std::filesystem::path& config_path) {
    const std::filesystem::path absolute_path =
        std::filesystem::absolute(config_path).lexically_normal();
    const JsonValue root_value =
        JsonParser(read_utf8_text(absolute_path)).parse();
    const auto& root = require_object(root_value, "$");
    constexpr std::array allowed{
        std::string_view("schema_version"),
        std::string_view("input"),
        std::string_view("acquisition"),
        std::string_view("image_grid"),
        std::string_view("receive"),
        std::string_view("tgc"),
        std::string_view("bandpass"),
        std::string_view("svd"),
        std::string_view("output")};
    reject_unknown_fields(root, allowed, "$");

    const std::size_t schema_version = require_size(
        require_member(root, "schema_version", "$"), "$.schema_version");
    if (schema_version != 1) {
        contract_error(
            "$.schema_version",
            std::to_string(schema_version),
            "supported version 1");
    }

    PipelineConfig config;
    config.source_path = absolute_path;
    config.schema_version = 1;
    config.input =
        parse_input(require_member(root, "input", "$"), absolute_path);
    config.acquisition = parse_acquisition(
        require_member(root, "acquisition", "$"), config.input.sequences);
    config.image_grid =
        parse_image_grid(require_member(root, "image_grid", "$"));
    config.receive = parse_receive(require_member(root, "receive", "$"));
    config.tgc = parse_tgc(require_member(root, "tgc", "$"));
    config.bandpass =
        parse_bandpass(require_member(root, "bandpass", "$"), absolute_path);
    config.svd =
        parse_svd(require_member(root, "svd", "$"), config.input.frame_count);
    config.output =
        parse_output(require_member(root, "output", "$"), absolute_path);
    return config;
}

}  // namespace carotid_doppler
