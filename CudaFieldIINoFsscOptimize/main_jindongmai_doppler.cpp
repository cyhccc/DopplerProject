#include "beamformer.cuh"
#include "blockwise_svd.cuh"
#include "config.cuh"
#include "data_preprocess_gpu_v2.cuh"
#include "motion_correction.h"
#include "PARAMs.h"
#include "rf_bandpass_filter.cuh"
#include "rsvd_filter.cuh"

#include <cublas_v2.h>
#include <cuComplex.h>
#include <cusolverDn.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

PARAMs pc;

namespace
{
	constexpr int kChunkFrames = 32;
	constexpr int kSeq = 5;
	constexpr int kChannels = 128;
	constexpr int kSamples = 2816;
	constexpr int kNx = 256;
	constexpr int kNz = 512;
	constexpr int kDefaultCutoffLowRank = 30;
	constexpr int kDefaultCutoffHighRank = 512;
	constexpr float kPi = 3.14159265358979323846f;

	const std::filesystem::path kDefaultInput =
		"E:/Code/dllconfig/linear_array/linear_arrays_plane_wave/linear_arrays_plane_wave/jindongmai_rf_middle512.bin";

	struct JsonValue
	{
		enum class Type
		{
			Null,
			Bool,
			Number,
			String,
			Array,
			Object
		};

		Type type = Type::Null;
		bool boolean = false;
		double number = 0.0;
		std::string string;
		std::vector<JsonValue> array;
		std::vector<std::pair<std::string, JsonValue>> object;

		const JsonValue* Find(const std::string& key) const
		{
			if (type != Type::Object)
			{
				return nullptr;
			}
			for (const auto& item : object)
			{
				if (item.first == key)
				{
					return &item.second;
				}
			}
			return nullptr;
		}
	};

	class JsonParser
	{
	public:
		explicit JsonParser(std::string text)
			: text_(std::move(text))
		{
		}

		JsonValue Parse()
		{
			SkipWhitespace();
			JsonValue value = ParseValue();
			SkipWhitespace();
			if (pos_ != text_.size())
			{
				throw std::runtime_error("Unexpected trailing characters in JSON.");
			}
			return value;
		}

	private:
		std::string text_;
		size_t pos_ = 0;

		void SkipWhitespace()
		{
			while (pos_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[pos_])) != 0)
			{
				++pos_;
			}
		}

		char Peek() const
		{
			return pos_ < text_.size() ? text_[pos_] : '\0';
		}

		char Take()
		{
			if (pos_ >= text_.size())
			{
				throw std::runtime_error("Unexpected end of JSON.");
			}
			return text_[pos_++];
		}

		void Expect(char expected)
		{
			const char got = Take();
			if (got != expected)
			{
				std::ostringstream ss;
				ss << "Expected '" << expected << "' in JSON, got '" << got << "'.";
				throw std::runtime_error(ss.str());
			}
		}

		JsonValue ParseValue()
		{
			SkipWhitespace();
			const char c = Peek();
			if (c == '{') { return ParseObject(); }
			if (c == '[') { return ParseArray(); }
			if (c == '"')
			{
				JsonValue value;
				value.type = JsonValue::Type::String;
				value.string = ParseString();
				return value;
			}
			if (c == '-' || (c >= '0' && c <= '9')) { return ParseNumber(); }
			if (MatchLiteral("true"))
			{
				JsonValue value;
				value.type = JsonValue::Type::Bool;
				value.boolean = true;
				return value;
			}
			if (MatchLiteral("false"))
			{
				JsonValue value;
				value.type = JsonValue::Type::Bool;
				value.boolean = false;
				return value;
			}
			if (MatchLiteral("null"))
			{
				return JsonValue{};
			}
			throw std::runtime_error("Unsupported JSON value.");
		}

		bool MatchLiteral(const char* literal)
		{
			const size_t len = std::char_traits<char>::length(literal);
			if (text_.compare(pos_, len, literal) == 0)
			{
				pos_ += len;
				return true;
			}
			return false;
		}

		std::string ParseString()
		{
			Expect('"');
			std::string out;
			while (true)
			{
				const char c = Take();
				if (c == '"')
				{
					return out;
				}
				if (c != '\\')
				{
					out.push_back(c);
					continue;
				}

				const char escaped = Take();
				switch (escaped)
				{
				case '"': out.push_back('"'); break;
				case '\\': out.push_back('\\'); break;
				case '/': out.push_back('/'); break;
				case 'b': out.push_back('\b'); break;
				case 'f': out.push_back('\f'); break;
				case 'n': out.push_back('\n'); break;
				case 'r': out.push_back('\r'); break;
				case 't': out.push_back('\t'); break;
				default:
					throw std::runtime_error("Unsupported JSON string escape.");
				}
			}
		}

		JsonValue ParseNumber()
		{
			const size_t start = pos_;
			if (Peek() == '-') { ++pos_; }
			while (std::isdigit(static_cast<unsigned char>(Peek())) != 0) { ++pos_; }
			if (Peek() == '.')
			{
				++pos_;
				while (std::isdigit(static_cast<unsigned char>(Peek())) != 0) { ++pos_; }
			}
			if (Peek() == 'e' || Peek() == 'E')
			{
				++pos_;
				if (Peek() == '+' || Peek() == '-') { ++pos_; }
				while (std::isdigit(static_cast<unsigned char>(Peek())) != 0) { ++pos_; }
			}
			JsonValue value;
			value.type = JsonValue::Type::Number;
			value.number = std::stod(text_.substr(start, pos_ - start));
			return value;
		}

		JsonValue ParseArray()
		{
			JsonValue value;
			value.type = JsonValue::Type::Array;
			Expect('[');
			SkipWhitespace();
			if (Peek() == ']')
			{
				++pos_;
				return value;
			}
			while (true)
			{
				value.array.push_back(ParseValue());
				SkipWhitespace();
				const char c = Take();
				if (c == ']') { return value; }
				if (c != ',')
				{
					throw std::runtime_error("Expected ',' or ']' in JSON array.");
				}
			}
		}

		JsonValue ParseObject()
		{
			JsonValue value;
			value.type = JsonValue::Type::Object;
			Expect('{');
			SkipWhitespace();
			if (Peek() == '}')
			{
				++pos_;
				return value;
			}
			while (true)
			{
				SkipWhitespace();
				const std::string key = ParseString();
				SkipWhitespace();
				Expect(':');
				value.object.emplace_back(key, ParseValue());
				SkipWhitespace();
				const char c = Take();
				if (c == '}') { return value; }
				if (c != ',')
				{
					throw std::runtime_error("Expected ',' or '}' in JSON object.");
				}
			}
		}
	};

	struct DopplerRunConfig
	{
		std::filesystem::path config_path;
		std::filesystem::path input_path = kDefaultInput;
		std::filesystem::path output_dir = "outputs/jindongmai_svd030";
		int seq = kSeq;
		int channels = kChannels;
		int samples = kSamples;
		float fs = 30.0e6f;
		float f0 = 5.0e6f;
		float sound_speed = 1480.0f;
		float pitch = 0.3e-3f;
		int img_nx = kNx;
		int img_nz = kNz;
		float x_start = -20.0e-3f;
		float x_end = 20.0e-3f;
		float z_start = 0.0f;
		float z_end = 70.0e-3f;
		float depth = 70.0e-3f;
		float F_number = 1.7f;
		float minimum_aperture = 1.0e-3f;
		int filter_length = 129;
		int show_range = 60;
		float cycle = 2.0f;
		std::vector<float> angle_list_deg = { -6.0f, -3.0f, 0.0f, 3.0f, 6.0f };
		float tx_delay_offset_samples = 113.0f;
		std::string channel_order = "ascending_x";
		std::string tx_reference_mode = "aligned_reference";
		bool svd_auto_ssm = false;
		int cutoff_low = kDefaultCutoffLowRank;
		int cutoff_high = kDefaultCutoffHighRank;
		int ssm_min_tissue = 5;
		int ssm_min_flow = 20;
		int ssm_max_components = 0;
		bool bandpass_enabled = false;
		std::filesystem::path bandpass_filter_path;
		int bandpass_taps = 129;
		bool motion_enabled = false;
		int motion_search_radius_x = 8;
		int motion_search_radius_z = 6;
		float motion_roi_x_start_m = -20.0e-3f;
		float motion_roi_x_end_m = 20.0e-3f;
		float motion_roi_z_start_m = 2.0e-3f;
		float motion_roi_z_end_m = 25.0e-3f;
		int motion_stride_x = 4;
		int motion_stride_z = 4;
		int motion_smooth_radius = 2;
		bool block_svd_enabled = false;
		int block_svd_width = 32;
		int block_svd_height = 32;
		float display_db_min = -40.0f;
		float display_db_max = 0.0f;
	};

	struct BlockSvdSummary
	{
		bool enabled = false;
		int block_width = 0;
		int block_height = 0;
		int block_count = 0;
		double gather_s = 0.0;
		double covariance_s = 0.0;
		double eig_s = 0.0;
		double projection_s = 0.0;
		double filter_s = 0.0;
		double scatter_s = 0.0;
	};

	const char* CublasStatusName(cublasStatus_t status)
	{
		switch (status)
		{
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

	void CheckCublas(cublasStatus_t status, const char* file, int line)
	{
		if (status != CUBLAS_STATUS_SUCCESS)
		{
			std::ostringstream ss;
			ss << "cuBLAS error " << CublasStatusName(status) << " at " << file << ":" << line;
			throw std::runtime_error(ss.str());
		}
	}

	void CheckCusolver(cusolverStatus_t status, const char* file, int line)
	{
		if (status != CUSOLVER_STATUS_SUCCESS)
		{
			std::ostringstream ss;
			ss << "cuSOLVER error code " << static_cast<int>(status) << " at " << file << ":" << line;
			throw std::runtime_error(ss.str());
		}
	}

#define CUBLAS_CHECK(call) CheckCublas((call), __FILE__, __LINE__)
#define CUSOLVER_CHECK(call) CheckCusolver((call), __FILE__, __LINE__)

	double SecondsSince(const std::chrono::high_resolution_clock::time_point& start)
	{
		return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
	}

	const JsonValue* FindNumberLike(const JsonValue& object, const std::string& key)
	{
		const JsonValue* value = object.Find(key);
		if (value == nullptr)
		{
			return nullptr;
		}
		if (value->type != JsonValue::Type::Number)
		{
			throw std::runtime_error("JSON field '" + key + "' must be a number.");
		}
		return value;
	}

	void ReadStringField(const JsonValue& object, const std::string& key, std::filesystem::path& out)
	{
		const JsonValue* value = object.Find(key);
		if (value == nullptr)
		{
			return;
		}
		if (value->type != JsonValue::Type::String)
		{
			throw std::runtime_error("JSON field '" + key + "' must be a string.");
		}
		out = value->string;
	}

	void ReadStringField(const JsonValue& object, const std::string& key, std::string& out)
	{
		const JsonValue* value = object.Find(key);
		if (value == nullptr)
		{
			return;
		}
		if (value->type != JsonValue::Type::String)
		{
			throw std::runtime_error("JSON field '" + key + "' must be a string.");
		}
		out = value->string;
	}

	void ReadIntField(const JsonValue& object, const std::string& key, int& out)
	{
		const JsonValue* value = FindNumberLike(object, key);
		if (value == nullptr)
		{
			return;
		}
		const double rounded = std::round(value->number);
		if (std::abs(value->number - rounded) > 1.0e-6)
		{
			throw std::runtime_error("JSON field '" + key + "' must be an integer.");
		}
		out = static_cast<int>(rounded);
	}

	void ReadFloatField(const JsonValue& object, const std::string& key, float& out)
	{
		const JsonValue* value = FindNumberLike(object, key);
		if (value == nullptr)
		{
			return;
		}
		out = static_cast<float>(value->number);
	}

	void ReadBoolField(const JsonValue& object, const std::string& key, bool& out)
	{
		const JsonValue* value = object.Find(key);
		if (value == nullptr)
		{
			return;
		}
		if (value->type != JsonValue::Type::Bool)
		{
			throw std::runtime_error("JSON field '" + key + "' must be a boolean.");
		}
		out = value->boolean;
	}

	void ReadFloatArrayField(const JsonValue& object, const std::string& key, std::vector<float>& out)
	{
		const JsonValue* value = object.Find(key);
		if (value == nullptr)
		{
			return;
		}
		if (value->type != JsonValue::Type::Array)
		{
			throw std::runtime_error("JSON field '" + key + "' must be a number array.");
		}
		std::vector<float> values;
		values.reserve(value->array.size());
		for (const JsonValue& item : value->array)
		{
			if (item.type != JsonValue::Type::Number)
			{
				throw std::runtime_error("JSON field '" + key + "' must contain only numbers.");
			}
			values.push_back(static_cast<float>(item.number));
		}
		out = std::move(values);
	}

	std::string AngleListToString(const std::vector<float>& angles)
	{
		std::ostringstream ss;
		for (size_t i = 0; i < angles.size(); ++i)
		{
			if (i != 0)
			{
				ss << ",";
			}
			ss << angles[i];
		}
		return ss.str();
	}

	bool HasJsonExtension(const std::filesystem::path& path)
	{
		std::string ext = path.extension().string();
		std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
			return static_cast<char>(std::tolower(c));
		});
		return ext == ".json";
	}

	std::filesystem::path ResolveRelativeToConfig(
		const std::filesystem::path& config_path,
		const std::filesystem::path& path)
	{
		if (path.empty() || path.is_absolute() || config_path.empty())
		{
			return path;
		}
		return config_path.parent_path() / path;
	}

	std::vector<float> ReadFloat32File(const std::filesystem::path& path)
	{
		if (!std::filesystem::exists(path))
		{
			throw std::runtime_error("Missing float32 file: " + path.string());
		}
		const size_t bytes = static_cast<size_t>(std::filesystem::file_size(path));
		if (bytes == 0 || (bytes % sizeof(float)) != 0)
		{
			throw std::runtime_error("Float32 file size is invalid: " + path.string());
		}
		std::vector<float> values(bytes / sizeof(float));
		std::ifstream file(path, std::ios::binary);
		if (!file.is_open())
		{
			throw std::runtime_error("Cannot open float32 file: " + path.string());
		}
		file.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(bytes));
		if (!file)
		{
			throw std::runtime_error("Failed to read float32 file: " + path.string());
		}
		return values;
	}

	int CoordinateToIndex(float value, float start, float step, int limit)
	{
		if (limit <= 1 || step == 0.0f)
		{
			return 0;
		}
		const int idx = static_cast<int>(std::round((value - start) / step));
		return std::clamp(idx, 0, limit - 1);
	}

	MotionCorrectionConfig BuildMotionCorrectionConfig(const DopplerRunConfig& config)
	{
		MotionCorrectionConfig motion{};
		motion.enabled = config.motion_enabled;
		motion.search_radius_x = config.motion_search_radius_x;
		motion.search_radius_z = config.motion_search_radius_z;
		motion.stride_x = config.motion_stride_x;
		motion.stride_z = config.motion_stride_z;
		motion.smooth_radius = config.motion_smooth_radius;

		const float pixel_dx = (config.img_nx > 1)
			? (config.x_end - config.x_start) / static_cast<float>(config.img_nx - 1)
			: 1.0f;
		const float pixel_dz = (config.img_nz > 1)
			? (config.z_end - config.z_start) / static_cast<float>(config.img_nz - 1)
			: 1.0f;
		motion.roi_x0 = CoordinateToIndex(config.motion_roi_x_start_m, config.x_start, pixel_dx, config.img_nx);
		motion.roi_x1 = CoordinateToIndex(config.motion_roi_x_end_m, config.x_start, pixel_dx, config.img_nx);
		motion.roi_z0 = CoordinateToIndex(config.motion_roi_z_start_m, config.z_start, pixel_dz, config.img_nz);
		motion.roi_z1 = CoordinateToIndex(config.motion_roi_z_end_m, config.z_start, pixel_dz, config.img_nz);
		if (motion.roi_x1 < motion.roi_x0) { std::swap(motion.roi_x0, motion.roi_x1); }
		if (motion.roi_z1 < motion.roi_z0) { std::swap(motion.roi_z0, motion.roi_z1); }
		return motion;
	}

	void WriteMotionShiftCsv(
		const std::filesystem::path& path,
		const MotionCorrectionSummary& summary)
	{
		std::ofstream out(path);
		if (!out.is_open())
		{
			throw std::runtime_error("Cannot write motion shift CSV: " + path.string());
		}
		out << "frame,raw_dx_px,raw_dz_px,raw_score,smoothed_dx_px,smoothed_dz_px,smoothed_score\n";
		const size_t count = std::min(summary.raw_shifts.size(), summary.smoothed_shifts.size());
		for (size_t i = 0; i < count; ++i)
		{
			const MotionShift& raw = summary.raw_shifts[i];
			const MotionShift& smoothed = summary.smoothed_shifts[i];
			out << i << ","
				<< raw.dx << "," << raw.dz << "," << raw.score << ","
				<< smoothed.dx << "," << smoothed.dz << "," << smoothed.score << "\n";
		}
	}

	void ValidateConfig(const DopplerRunConfig& config)
	{
		if (config.seq <= 0 || config.channels <= 0 || config.samples <= 0)
		{
			throw std::runtime_error("Config dimensions seq/channels/datalen must be positive.");
		}
		if (config.img_nx <= 0 || config.img_nz <= 0)
		{
			throw std::runtime_error("Config image dimensions img_nx/img_nz must be positive.");
		}
		if (config.angle_list_deg.size() != static_cast<size_t>(config.seq))
		{
			throw std::runtime_error("Config angle_list length must equal seq.");
		}
		if (config.seq > 2)
		{
			const float step = config.angle_list_deg[1] - config.angle_list_deg[0];
			for (int i = 2; i < config.seq; ++i)
			{
				const float current_step = config.angle_list_deg[static_cast<size_t>(i)] -
					config.angle_list_deg[static_cast<size_t>(i - 1)];
				if (std::abs(current_step - step) > 1.0e-4f)
				{
					throw std::runtime_error("Current beamformer supports only evenly spaced angle_list values.");
				}
			}
		}
		if (!config.svd_auto_ssm && (config.cutoff_low < 1 || config.cutoff_high < config.cutoff_low))
		{
			throw std::runtime_error("Invalid SVD cutoff in config. Expected 1 <= cutoff_low <= cutoff_high.");
		}
		if (config.ssm_min_tissue < 0 || config.ssm_min_flow < 0)
		{
			throw std::runtime_error("SSM component constraints must be non-negative.");
		}
		if (!(config.display_db_min < config.display_db_max))
		{
			throw std::runtime_error("display_db_min must be smaller than display_db_max.");
		}
		if (config.channel_order != "ascending_x" && config.channel_order != "descending_x")
		{
			throw std::runtime_error("channel_order must be 'ascending_x' or 'descending_x'.");
		}
		if (config.tx_reference_mode != "aligned_reference" &&
			config.tx_reference_mode != "aperture_edge")
		{
			throw std::runtime_error(
				"tx_reference_mode must be 'aligned_reference' or 'aperture_edge'.");
		}
		if (config.bandpass_enabled)
		{
			if (config.bandpass_filter_path.empty())
			{
				throw std::runtime_error("bandpass.filter_path is required when bandpass.enabled is true.");
			}
			if (config.bandpass_taps <= 0 || (config.bandpass_taps % 2) == 0)
			{
				throw std::runtime_error("bandpass.taps must be a positive odd integer.");
			}
		}
		if (config.motion_enabled)
		{
			if (config.motion_search_radius_x < 0 || config.motion_search_radius_z < 0)
			{
				throw std::runtime_error("motion_correction search radii must be non-negative.");
			}
			if (config.motion_stride_x <= 0 || config.motion_stride_z <= 0)
			{
				throw std::runtime_error("motion_correction strides must be positive.");
			}
			if (config.motion_smooth_radius < 0)
			{
				throw std::runtime_error("motion_correction smooth_radius must be non-negative.");
			}
			(void)BuildMotionCorrectionConfig(config);
		}
		if (config.block_svd_enabled)
		{
			if (config.block_svd_width <= 0 || config.block_svd_height <= 0)
			{
				throw std::runtime_error("block_svd block_width/block_height must be positive.");
			}
			if (config.block_svd_width > config.img_nx || config.block_svd_height > config.img_nz)
			{
				throw std::runtime_error("block_svd block size must not exceed image dimensions.");
			}
		}
	}

	DopplerRunConfig LoadConfigJson(const std::filesystem::path& config_path)
	{
		std::ifstream file(config_path);
		if (!file.is_open())
		{
			throw std::runtime_error("Cannot open config JSON: " + config_path.string());
		}
		std::ostringstream buffer;
		buffer << file.rdbuf();
		JsonValue root = JsonParser(buffer.str()).Parse();
		if (root.type != JsonValue::Type::Object)
		{
			throw std::runtime_error("Config JSON root must be an object.");
		}

		DopplerRunConfig config;
		config.config_path = config_path;
		ReadStringField(root, "input_path", config.input_path);
		ReadStringField(root, "output_dir", config.output_dir);
		ReadIntField(root, "seq", config.seq);
		ReadIntField(root, "channels", config.channels);
		ReadIntField(root, "datalen", config.samples);
		ReadFloatField(root, "fs", config.fs);
		ReadFloatField(root, "f0", config.f0);
		ReadFloatField(root, "sound_speed", config.sound_speed);
		ReadFloatField(root, "pitch", config.pitch);
		ReadIntField(root, "img_nx", config.img_nx);
		ReadIntField(root, "img_nz", config.img_nz);
		ReadFloatField(root, "x_start", config.x_start);
		ReadFloatField(root, "x_end", config.x_end);
		ReadFloatField(root, "z_start", config.z_start);
		ReadFloatField(root, "z_end", config.z_end);
		config.depth = config.z_end;
		ReadFloatField(root, "depth", config.depth);
		ReadFloatField(root, "F_number", config.F_number);
		ReadFloatField(root, "minimum_aperture", config.minimum_aperture);
		ReadIntField(root, "filterLength", config.filter_length);
		ReadIntField(root, "show_range", config.show_range);
		ReadFloatField(root, "cycle", config.cycle);
		ReadFloatArrayField(root, "angle_list", config.angle_list_deg);
		ReadFloatField(root, "tx_delay_offset_samples", config.tx_delay_offset_samples);
		ReadStringField(root, "channel_order", config.channel_order);
		ReadStringField(root, "tx_reference_mode", config.tx_reference_mode);
		ReadFloatField(root, "display_db_min", config.display_db_min);
		ReadFloatField(root, "display_db_max", config.display_db_max);

		if (const JsonValue* svd = root.Find("svd"))
		{
			if (svd->type != JsonValue::Type::Object)
			{
				throw std::runtime_error("JSON field 'svd' must be an object.");
			}
			ReadBoolField(*svd, "auto_ssm", config.svd_auto_ssm);
			ReadIntField(*svd, "cutoff_low", config.cutoff_low);
			ReadIntField(*svd, "cutoff_high", config.cutoff_high);
			ReadIntField(*svd, "ssm_min_tissue", config.ssm_min_tissue);
			ReadIntField(*svd, "ssm_min_flow", config.ssm_min_flow);
			ReadIntField(*svd, "ssm_max_components", config.ssm_max_components);
		}

		if (const JsonValue* bandpass = root.Find("bandpass"))
		{
			if (bandpass->type != JsonValue::Type::Object)
			{
				throw std::runtime_error("JSON field 'bandpass' must be an object.");
			}
			ReadBoolField(*bandpass, "enabled", config.bandpass_enabled);
			ReadStringField(*bandpass, "filter_path", config.bandpass_filter_path);
			ReadIntField(*bandpass, "taps", config.bandpass_taps);
			config.bandpass_filter_path = ResolveRelativeToConfig(config.config_path, config.bandpass_filter_path);
		}

		if (const JsonValue* motion = root.Find("motion_correction"))
		{
			if (motion->type != JsonValue::Type::Object)
			{
				throw std::runtime_error("JSON field 'motion_correction' must be an object.");
			}
			ReadBoolField(*motion, "enabled", config.motion_enabled);
			ReadIntField(*motion, "search_radius_x", config.motion_search_radius_x);
			ReadIntField(*motion, "search_radius_z", config.motion_search_radius_z);
			ReadFloatField(*motion, "roi_x_start", config.motion_roi_x_start_m);
			ReadFloatField(*motion, "roi_x_end", config.motion_roi_x_end_m);
			ReadFloatField(*motion, "roi_z_start", config.motion_roi_z_start_m);
			ReadFloatField(*motion, "roi_z_end", config.motion_roi_z_end_m);
			ReadIntField(*motion, "stride_x", config.motion_stride_x);
			ReadIntField(*motion, "stride_z", config.motion_stride_z);
			ReadIntField(*motion, "smooth_radius", config.motion_smooth_radius);
		}

		if (const JsonValue* block_svd = root.Find("block_svd"))
		{
			if (block_svd->type != JsonValue::Type::Object)
			{
				throw std::runtime_error("JSON field 'block_svd' must be an object.");
			}
			ReadBoolField(*block_svd, "enabled", config.block_svd_enabled);
			ReadIntField(*block_svd, "block_width", config.block_svd_width);
			ReadIntField(*block_svd, "block_height", config.block_svd_height);
		}

		ValidateConfig(config);
		return config;
	}

	DopplerRunConfig LoadConfigFromArgs(int argc, char** argv)
	{
		if (argc >= 2 && HasJsonExtension(std::filesystem::path(argv[1])))
		{
			return LoadConfigJson(argv[1]);
		}

		DopplerRunConfig config;
		config.input_path = (argc >= 2) ? std::filesystem::path(argv[1]) : kDefaultInput;
		config.output_dir = (argc >= 3)
			? std::filesystem::path(argv[2])
			: std::filesystem::path("outputs/jindongmai_svd030");
		config.svd_auto_ssm = argc < 4;
		config.cutoff_low = (argc >= 4) ? std::stoi(argv[3]) : kDefaultCutoffLowRank;
		config.cutoff_high = (argc >= 5) ? std::stoi(argv[4]) : kDefaultCutoffHighRank;
		ValidateConfig(config);
		return config;
	}

	void ConfigureJindongmai(const DopplerRunConfig& config)
	{
		pc.seq = config.seq;
		pc.datalen = config.samples;
		pc.channels = config.channels;
		pc.f0 = config.f0;
		pc.fs = config.fs;
		pc.sound_speed = config.sound_speed;
		pc.pitch = config.pitch;
		pc.img_nx = config.img_nx;
		pc.img_nz = config.img_nz;
		pc.x_start = config.x_start;
		pc.x_end = config.x_end;
		pc.z_start = config.z_start;
		pc.z_end = config.z_end;
		pc.depth = config.depth;
		pc.F_number = config.F_number;
		pc.minimum_aperture = config.minimum_aperture;
		pc.filterLength = config.filter_length;
		pc.show_range = config.show_range;
		pc.cycle = config.cycle;
		pc.use_auto_angle_span = false;
		pc.init_angle = config.angle_list_deg.front() * (kPi / 180.0f);
		pc.final_angle = config.angle_list_deg.back() * (kPi / 180.0f);
		pc.tx_delay_offset_samples = config.tx_delay_offset_samples;
		pc.tx_reference_aperture_edge = config.tx_reference_mode == "aperture_edge";
		pc.UpdateParamter();
	}

	size_t RfElementsPerFrame(const DopplerRunConfig& config)
	{
		return static_cast<size_t>(config.seq) * config.samples * config.channels;
	}

	int ValidateInputFileAndGetFrames(const std::filesystem::path& input_path, const DopplerRunConfig& config)
	{
		if (!std::filesystem::exists(input_path))
		{
			throw std::runtime_error("Missing input bin: " + input_path.string());
		}

		const size_t frame_bytes = RfElementsPerFrame(config) * sizeof(std::int16_t);
		const size_t actual_bytes = static_cast<size_t>(std::filesystem::file_size(input_path));
		if (actual_bytes == 0 || actual_bytes % frame_bytes != 0)
		{
			std::ostringstream ss;
			ss << "Unexpected input size for " << input_path.string()
				<< ". File bytes must be a multiple of one frame: "
				<< config.seq << "x" << config.samples << "x" << config.channels
				<< " int16, frame_bytes=" << frame_bytes
				<< ", actual_bytes=" << actual_bytes << ".";
			throw std::runtime_error(ss.str());
		}
		const size_t frames = actual_bytes / frame_bytes;
		if (frames > static_cast<size_t>(std::numeric_limits<int>::max()))
		{
			throw std::runtime_error("Too many frames in input file.");
		}
		return static_cast<int>(frames);
	}

	void ReadRfChunk(
		std::ifstream& file,
		int start_frame,
		int chunk_frames,
		std::vector<float>& chunk,
		const DopplerRunConfig& config)
	{
		const size_t frame_elements = RfElementsPerFrame(config);
		const size_t chunk_elements = frame_elements * static_cast<size_t>(chunk_frames);
		const size_t byte_offset = frame_elements * static_cast<size_t>(start_frame) * sizeof(std::int16_t);
		const size_t byte_count = chunk_elements * sizeof(std::int16_t);
		std::vector<std::int16_t> short_chunk(chunk_elements);
		chunk.resize(chunk_elements);
		const bool reverse_channels = config.channel_order == "descending_x";
		file.seekg(static_cast<std::streamoff>(byte_offset), std::ios::beg);
		file.read(reinterpret_cast<char*>(short_chunk.data()), static_cast<std::streamsize>(byte_count));
		if (!file)
		{
			throw std::runtime_error("Failed to read RF chunk starting at frame " + std::to_string(start_frame));
		}
		for (int frame = 0; frame < chunk_frames; ++frame)
		{
			for (int seq = 0; seq < config.seq; ++seq)
			{
				for (int sample = 0; sample < config.samples; ++sample)
				{
					for (int channel = 0; channel < config.channels; ++channel)
					{
						const int canonical_channel = reverse_channels
							? config.channels - 1 - channel
							: channel;
						const size_t source_idx =
							((static_cast<size_t>(frame) * config.seq + seq) * config.samples + sample) *
							config.channels + channel;
						const size_t target_idx =
							((static_cast<size_t>(frame) * config.seq + seq) * config.channels + canonical_channel) *
							config.samples + sample;
						chunk[target_idx] = static_cast<float>(short_chunk[source_idx]);
					}
				}
			}
		}
	}

	void WriteFloatBin(const std::filesystem::path& path, const std::vector<float>& data)
	{
		std::ofstream file(path, std::ios::binary);
		if (!file.is_open())
		{
			throw std::runtime_error("Cannot write output file: " + path.string());
		}
		file.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)));
		if (!file)
		{
			throw std::runtime_error("Output write failed: " + path.string());
		}
	}

	void BeamformAllFramesChunked(
		const std::filesystem::path& input_path,
		int frame_count,
		float2* d_iq_cache,
		double& read_seconds,
		double& bandpass_seconds,
		double& rf_to_iq_seconds,
		double& beamform_seconds,
		const DopplerRunConfig& config)
	{
		const int pixels = pc.img_nx * pc.img_nz;
		const size_t chunk_elements = static_cast<size_t>(kChunkFrames) * config.seq * config.channels * config.samples;
		const size_t chunk_bytes = chunk_elements * sizeof(float);
		const size_t chunk_iq_bytes = chunk_elements * sizeof(float2);
		const size_t chunk_cache_elements = static_cast<size_t>(pixels) * kChunkFrames;
		const size_t chunk_cache_bytes = chunk_cache_elements * sizeof(float2);

		std::ifstream file(input_path, std::ios::binary);
		if (!file.is_open())
		{
			throw std::runtime_error("Cannot open input file: " + input_path.string());
		}

		float* d_rf_chunk = nullptr;
		float* d_rf_bandpass_chunk = nullptr;
		float* d_bandpass_coefficients = nullptr;
		float2* d_iq_chunk = nullptr;
		float2* d_iq_cache_chunk = nullptr;

		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rf_chunk), chunk_bytes));
		if (config.bandpass_enabled)
		{
			const std::vector<float> coefficients = ReadFloat32File(config.bandpass_filter_path);
			if (coefficients.size() != static_cast<size_t>(config.bandpass_taps))
			{
				std::ostringstream ss;
				ss << "Bandpass coefficient count mismatch. Expected " << config.bandpass_taps
					<< ", got " << coefficients.size()
					<< " from " << config.bandpass_filter_path.string() << ".";
				throw std::runtime_error(ss.str());
			}
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rf_bandpass_chunk), chunk_bytes));
			CUDA_CHECK(cudaMalloc(
				reinterpret_cast<void**>(&d_bandpass_coefficients),
				coefficients.size() * sizeof(float)));
			CUDA_CHECK(cudaMemcpy(
				d_bandpass_coefficients,
				coefficients.data(),
				coefficients.size() * sizeof(float),
				cudaMemcpyHostToDevice));
		}
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_iq_chunk), chunk_iq_bytes));
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_iq_cache_chunk), chunk_cache_bytes));

		try
		{
			BEAMFORMER beamform;
			std::vector<float> h_chunk;

			for (int frame = 0; frame < frame_count; frame += kChunkFrames)
			{
				const int current_frames = std::min(kChunkFrames, frame_count - frame);
				const size_t current_elements =
					static_cast<size_t>(current_frames) * config.seq * config.channels * config.samples;
				const size_t current_bytes = current_elements * sizeof(float);
				const size_t current_cache_elements = static_cast<size_t>(pixels) * current_frames;
				const size_t current_cache_bytes = current_cache_elements * sizeof(float2);

				const auto read_start = std::chrono::high_resolution_clock::now();
				ReadRfChunk(file, frame, current_frames, h_chunk, config);
				read_seconds += SecondsSince(read_start);

				CUDA_CHECK(cudaMemcpy(d_rf_chunk, h_chunk.data(), current_bytes, cudaMemcpyHostToDevice));
				CUDA_CHECK(cudaDeviceSynchronize());

				const float* d_rf_for_iq = d_rf_chunk;
				if (config.bandpass_enabled)
				{
					const int traces = current_frames * config.seq * config.channels;
					const auto bandpass_start = std::chrono::high_resolution_clock::now();
					ApplyRfBandpassFilter(
						d_rf_chunk,
						d_rf_bandpass_chunk,
						d_bandpass_coefficients,
						traces,
						config.samples,
						config.bandpass_taps);
					CUDA_CHECK(cudaDeviceSynchronize());
					bandpass_seconds += SecondsSince(bandpass_start);
					d_rf_for_iq = d_rf_bandpass_chunk;
				}

				FieldIIRfToIqPreprocessor rf_to_iq(config.samples, config.channels, config.seq, current_frames);
				const auto rf_start = std::chrono::high_resolution_clock::now();
				rf_to_iq.RfToIq(d_rf_for_iq, d_iq_chunk);
				CUDA_CHECK(cudaDeviceSynchronize());
				rf_to_iq_seconds += SecondsSince(rf_start);

				const auto beam_start = std::chrono::high_resolution_clock::now();
				beamform.BeamformIQCacheSharedRectChannelOuterCore(d_iq_chunk, current_frames, d_iq_cache_chunk);
				CUDA_CHECK(cudaDeviceSynchronize());
				beamform_seconds += SecondsSince(beam_start);

				const size_t dst_offset = static_cast<size_t>(frame) * pixels;
				CUDA_CHECK(cudaMemcpy(
					d_iq_cache + dst_offset,
					d_iq_cache_chunk,
					current_cache_bytes,
					cudaMemcpyDeviceToDevice));

				std::cout << "[Chunk] frames " << frame << ".." << (frame + current_frames - 1) << " done\n";
			}

			CUDA_CHECK(cudaFree(d_iq_cache_chunk));
			d_iq_cache_chunk = nullptr;
			CUDA_CHECK(cudaFree(d_iq_chunk));
			d_iq_chunk = nullptr;
			if (d_bandpass_coefficients != nullptr)
			{
				CUDA_CHECK(cudaFree(d_bandpass_coefficients));
				d_bandpass_coefficients = nullptr;
			}
			if (d_rf_bandpass_chunk != nullptr)
			{
				CUDA_CHECK(cudaFree(d_rf_bandpass_chunk));
				d_rf_bandpass_chunk = nullptr;
			}
			CUDA_CHECK(cudaFree(d_rf_chunk));
			d_rf_chunk = nullptr;
		}
		catch (...)
		{
			if (d_iq_cache_chunk != nullptr) { cudaFree(d_iq_cache_chunk); }
			if (d_iq_chunk != nullptr) { cudaFree(d_iq_chunk); }
			if (d_bandpass_coefficients != nullptr) { cudaFree(d_bandpass_coefficients); }
			if (d_rf_bandpass_chunk != nullptr) { cudaFree(d_rf_bandpass_chunk); }
			if (d_rf_chunk != nullptr) { cudaFree(d_rf_chunk); }
			throw;
		}
	}

	void BuildSvdRankProjectionRows(
		cublasHandle_t cublas,
		cusolverDnHandle_t solver,
		float2* d_iq_cache,
		int row_count,
		cuComplex* d_projection,
		std::vector<float>& eigenvalues,
		int frame_count,
		int cutoff_low_rank,
		int cutoff_high_rank,
		double& covariance_seconds,
		double& eig_seconds,
		double& projection_seconds)
	{
		const int n = frame_count;
		const cuComplex alpha = make_cuFloatComplex(1.0f, 0.0f);
		const cuComplex beta = make_cuFloatComplex(0.0f, 0.0f);
		cuComplex* const d_x = reinterpret_cast<cuComplex*>(d_iq_cache);

		cuComplex* d_cov = nullptr;
		cuComplex* d_v_scaled = nullptr;
		float* d_w = nullptr;
		cuComplex* d_work = nullptr;
		int* d_info = nullptr;

		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_cov), static_cast<size_t>(n) * n * sizeof(cuComplex)));
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_v_scaled), static_cast<size_t>(n) * n * sizeof(cuComplex)));
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_w), static_cast<size_t>(n) * sizeof(float)));
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_info), sizeof(int)));

		try
		{
			const auto cov_start = std::chrono::high_resolution_clock::now();
			CUBLAS_CHECK(cublasCgemm(
				cublas,
				CUBLAS_OP_C,
				CUBLAS_OP_N,
				n,
				n,
				row_count,
				&alpha,
				d_x,
				row_count,
				d_x,
				row_count,
				&beta,
				d_cov,
				n));
			CUDA_CHECK(cudaDeviceSynchronize());
			covariance_seconds = SecondsSince(cov_start);

			int lwork = 0;
			CUSOLVER_CHECK(cusolverDnCheevd_bufferSize(
				solver,
				CUSOLVER_EIG_MODE_VECTOR,
				CUBLAS_FILL_MODE_UPPER,
				n,
				d_cov,
				n,
				d_w,
				&lwork));
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_work), static_cast<size_t>(lwork) * sizeof(cuComplex)));

			const auto eig_start = std::chrono::high_resolution_clock::now();
			CUSOLVER_CHECK(cusolverDnCheevd(
				solver,
				CUSOLVER_EIG_MODE_VECTOR,
				CUBLAS_FILL_MODE_UPPER,
				n,
				d_cov,
				n,
				d_w,
				d_work,
				lwork,
				d_info));
			CUDA_CHECK(cudaDeviceSynchronize());
			eig_seconds = SecondsSince(eig_start);

			int info = 0;
			CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
			if (info != 0)
			{
				throw std::runtime_error("cusolverDnCheevd failed with info=" + std::to_string(info));
			}

			eigenvalues.resize(n);
			CUDA_CHECK(cudaMemcpy(eigenvalues.data(), d_w, static_cast<size_t>(n) * sizeof(float), cudaMemcpyDeviceToHost));

			std::vector<cuComplex> h_v(static_cast<size_t>(n) * n);
			CUDA_CHECK(cudaMemcpy(h_v.data(), d_cov, h_v.size() * sizeof(cuComplex), cudaMemcpyDeviceToHost));
			for (int col = 0; col < n; ++col)
			{
				const int svd_rank_one_based = n - col;
				const bool keep = svd_rank_one_based >= cutoff_low_rank && svd_rank_one_based <= cutoff_high_rank;
				if (!keep)
				{
					for (int row = 0; row < n; ++row)
					{
						h_v[static_cast<size_t>(col) * n + row] = make_cuFloatComplex(0.0f, 0.0f);
					}
				}
			}
			CUDA_CHECK(cudaMemcpy(d_v_scaled, h_v.data(), h_v.size() * sizeof(cuComplex), cudaMemcpyHostToDevice));

			const auto projection_start = std::chrono::high_resolution_clock::now();
			CUBLAS_CHECK(cublasCgemm(
				cublas,
				CUBLAS_OP_N,
				CUBLAS_OP_C,
				n,
				n,
				n,
				&alpha,
				d_v_scaled,
				n,
				d_cov,
				n,
				&beta,
				d_projection,
				n));
			CUDA_CHECK(cudaDeviceSynchronize());
			projection_seconds = SecondsSince(projection_start);

			CUDA_CHECK(cudaFree(d_work));
			d_work = nullptr;
			CUDA_CHECK(cudaFree(d_info));
			d_info = nullptr;
			CUDA_CHECK(cudaFree(d_w));
			d_w = nullptr;
			CUDA_CHECK(cudaFree(d_v_scaled));
			d_v_scaled = nullptr;
			CUDA_CHECK(cudaFree(d_cov));
			d_cov = nullptr;
		}
		catch (...)
		{
			if (d_work != nullptr) { cudaFree(d_work); }
			if (d_info != nullptr) { cudaFree(d_info); }
			if (d_w != nullptr) { cudaFree(d_w); }
			if (d_v_scaled != nullptr) { cudaFree(d_v_scaled); }
			if (d_cov != nullptr) { cudaFree(d_cov); }
			throw;
		}
	}

	void BuildSvdRankProjection(
		cublasHandle_t cublas,
		cusolverDnHandle_t solver,
		float2* d_iq_cache,
		cuComplex* d_projection,
		std::vector<float>& eigenvalues,
		int frame_count,
		int cutoff_low_rank,
		int cutoff_high_rank,
		double& covariance_seconds,
		double& eig_seconds,
		double& projection_seconds)
	{
		BuildSvdRankProjectionRows(
			cublas,
			solver,
			d_iq_cache,
			pc.img_nx * pc.img_nz,
			d_projection,
			eigenvalues,
			frame_count,
			cutoff_low_rank,
			cutoff_high_rank,
			covariance_seconds,
			eig_seconds,
			projection_seconds);
	}

	void ApplyProjectionRows(
		cublasHandle_t cublas,
		float2* d_iq_cache,
		int row_count,
		const cuComplex* d_projection,
		float2* d_filtered,
		int frame_count,
		double& filter_seconds);

	void ApplyProjection(
		cublasHandle_t cublas,
		float2* d_iq_cache,
		const cuComplex* d_projection,
		float2* d_filtered,
		int frame_count,
		double& filter_seconds)
	{
		ApplyProjectionRows(
			cublas,
			d_iq_cache,
			pc.img_nx * pc.img_nz,
			d_projection,
			d_filtered,
			frame_count,
			filter_seconds);
	}

	void ApplyProjectionRows(
		cublasHandle_t cublas,
		float2* d_iq_cache,
		int row_count,
		const cuComplex* d_projection,
		float2* d_filtered,
		int frame_count,
		double& filter_seconds)
	{
		const int n = frame_count;
		const cuComplex alpha = make_cuFloatComplex(1.0f, 0.0f);
		const cuComplex beta = make_cuFloatComplex(0.0f, 0.0f);

		const auto filter_start = std::chrono::high_resolution_clock::now();
		CUBLAS_CHECK(cublasCgemm(
			cublas,
			CUBLAS_OP_N,
			CUBLAS_OP_N,
			row_count,
			n,
			n,
			&alpha,
			reinterpret_cast<const cuComplex*>(d_iq_cache),
			row_count,
			d_projection,
			n,
			&beta,
			reinterpret_cast<cuComplex*>(d_filtered),
			row_count));
		CUDA_CHECK(cudaDeviceSynchronize());
		filter_seconds = SecondsSince(filter_start);
	}

	BlockSvdSummary RunBlockwiseSvdFilter(
		cublasHandle_t cublas,
		cusolverDnHandle_t solver,
		float2* d_iq_cache,
		float2* d_filtered,
		cuComplex* d_projection,
		std::vector<float>& representative_eigenvalues,
		int frame_count,
		int cutoff_low_rank,
		int cutoff_high_rank,
		const DopplerRunConfig& config)
	{
		BlockSvdSummary summary{};
		summary.enabled = true;
		summary.block_width = config.block_svd_width;
		summary.block_height = config.block_svd_height;

		const int blocks_x = (pc.img_nx + config.block_svd_width - 1) / config.block_svd_width;
		const int blocks_z = (pc.img_nz + config.block_svd_height - 1) / config.block_svd_height;
		summary.block_count = blocks_x * blocks_z;
		const int max_block_pixels = config.block_svd_width * config.block_svd_height;
		const size_t max_block_elements = static_cast<size_t>(max_block_pixels) * frame_count;
		const size_t max_block_bytes = max_block_elements * sizeof(float2);

		float2* d_block_in = nullptr;
		float2* d_block_out = nullptr;
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_block_in), max_block_bytes));
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_block_out), max_block_bytes));

		try
		{
			int block_index = 0;
			std::vector<float> block_eigenvalues;
			for (int z0 = 0; z0 < pc.img_nz; z0 += config.block_svd_height)
			{
				const int block_h = std::min(config.block_svd_height, pc.img_nz - z0);
				for (int x0 = 0; x0 < pc.img_nx; x0 += config.block_svd_width)
				{
					const int block_w = std::min(config.block_svd_width, pc.img_nx - x0);
					const int block_pixels = block_w * block_h;
					++block_index;

					const auto gather_start = std::chrono::high_resolution_clock::now();
					GatherIqBlockFrameMajor(
						d_iq_cache,
						d_block_in,
						pc.img_nx,
						pc.img_nz,
						frame_count,
						x0,
						z0,
						block_w,
						block_h);
					CUDA_CHECK(cudaDeviceSynchronize());
					summary.gather_s += SecondsSince(gather_start);

					double block_covariance_s = 0.0;
					double block_eig_s = 0.0;
					double block_projection_s = 0.0;
					BuildSvdRankProjectionRows(
						cublas,
						solver,
						d_block_in,
						block_pixels,
						d_projection,
						block_eigenvalues,
						frame_count,
						cutoff_low_rank,
						cutoff_high_rank,
						block_covariance_s,
						block_eig_s,
						block_projection_s);
					summary.covariance_s += block_covariance_s;
					summary.eig_s += block_eig_s;
					summary.projection_s += block_projection_s;
					if (representative_eigenvalues.empty())
					{
						representative_eigenvalues = block_eigenvalues;
					}

					double block_filter_s = 0.0;
					ApplyProjectionRows(
						cublas,
						d_block_in,
						block_pixels,
						d_projection,
						d_block_out,
						frame_count,
						block_filter_s);
					summary.filter_s += block_filter_s;

					const auto scatter_start = std::chrono::high_resolution_clock::now();
					ScatterIqBlockFrameMajor(
						d_block_out,
						d_filtered,
						pc.img_nx,
						pc.img_nz,
						frame_count,
						x0,
						z0,
						block_w,
						block_h);
					CUDA_CHECK(cudaDeviceSynchronize());
					summary.scatter_s += SecondsSince(scatter_start);

					if (block_index == 1 || block_index == summary.block_count || (block_index % 8) == 0)
					{
						std::cout << "[BlockSVD] block " << block_index << "/" << summary.block_count
							<< " x=[" << x0 << "," << (x0 + block_w - 1) << "]"
							<< " z=[" << z0 << "," << (z0 + block_h - 1) << "]\n";
					}
				}
			}

			CUDA_CHECK(cudaFree(d_block_out));
			d_block_out = nullptr;
			CUDA_CHECK(cudaFree(d_block_in));
			d_block_in = nullptr;
		}
		catch (...)
		{
			if (d_block_out != nullptr) { cudaFree(d_block_out); }
			if (d_block_in != nullptr) { cudaFree(d_block_in); }
			throw;
		}

		return summary;
	}

	void ComputePowerAndDb(
		const std::vector<float2>& filtered,
		int frame_count,
		std::vector<float>& power,
		std::vector<float>& db,
		double& db_mean,
		double& db_std,
		double& display_dr)
	{
		const int pixels = pc.img_nx * pc.img_nz;
		power.assign(static_cast<size_t>(pixels), 0.0f);
		for (int frame = 0; frame < frame_count; ++frame)
		{
			const size_t frame_offset = static_cast<size_t>(frame) * pixels;
			for (int pixel = 0; pixel < pixels; ++pixel)
			{
				const float2 value = filtered[frame_offset + pixel];
				power[pixel] += value.x * value.x + value.y * value.y;
			}
		}

		const float max_power = *std::max_element(power.begin(), power.end());
		if (!(max_power > 0.0f) || !std::isfinite(max_power))
		{
			throw std::runtime_error("Invalid Power Doppler result: max power is not positive finite.");
		}

		db.resize(power.size());
		double sum = 0.0;
		for (size_t i = 0; i < power.size(); ++i)
		{
			const float normalized = std::max(power[i] / max_power, 1.0e-30f);
			db[i] = 10.0f * std::log10(normalized);
			sum += db[i];
		}
		db_mean = sum / static_cast<double>(db.size());

		double variance = 0.0;
		for (float value : db)
		{
			const double delta = static_cast<double>(value) - db_mean;
			variance += delta * delta;
		}
		db_std = std::sqrt(variance / static_cast<double>(db.size()));
		display_dr = 0.5 * db_std + std::abs(db_mean);
	}

	void WriteSummary(
		const std::filesystem::path& path,
		const DopplerRunConfig& config,
		const std::filesystem::path& input_path,
		const std::filesystem::path& power_path,
		const std::filesystem::path& db_path,
		const std::vector<float>& power,
		const std::vector<float>& db,
		const std::vector<float>& eigenvalues,
		const SsmCutoffResult& ssm_result,
		const MotionCorrectionSummary& motion_summary,
		const std::filesystem::path& motion_csv_path,
		const BlockSvdSummary& block_svd_summary,
		int frame_count,
		int cutoff_low_rank,
		int cutoff_high_rank,
		double read_s,
		double bandpass_s,
		double rf_to_iq_s,
		double beamform_s,
		double covariance_s,
		double eig_s,
		double projection_s,
		double filter_s,
		double db_mean,
		double db_std,
		double display_dr)
	{
		const auto power_minmax = std::minmax_element(power.begin(), power.end());
		const auto db_minmax = std::minmax_element(db.begin(), db.end());
		std::ofstream out(path);
		if (!out.is_open())
		{
			throw std::runtime_error("Cannot write summary: " + path.string());
		}

		out << std::setprecision(9)
			<< "mode=jindongmai_post_beamform_svd_power_doppler\n"
			<< "config=" << (config.config_path.empty() ? std::string("cli") : config.config_path.string()) << "\n"
			<< "input=" << input_path.string() << "\n"
			<< "input_layout=file=[frame][seq][sample][channel] int16, cuda_preprocess=[frame][seq][channel][sample] float32\n"
			<< "source_channel_order=" << config.channel_order << "\n"
			<< "internal_channel_order=ascending_x\n"
			<< "channel_reversed=" << (config.channel_order == "descending_x" ? "true" : "false") << "\n"
			<< "frames=" << frame_count << "\n"
			<< "seq=" << pc.seq << "\n"
			<< "channels=" << pc.channels << "\n"
			<< "samples=" << pc.datalen << "\n"
			<< "fs_hz=" << pc.fs << "\n"
			<< "f0_hz=" << pc.f0 << "\n"
			<< "sound_speed_mps=" << pc.sound_speed << "\n"
			<< "pitch_m=" << pc.pitch << "\n"
			<< "img_nx=" << pc.img_nx << "\n"
			<< "img_nz=" << pc.img_nz << "\n"
			<< "x_start_m=" << pc.x_start << "\n"
			<< "x_end_m=" << pc.x_end << "\n"
			<< "z_start_m=" << pc.z_start << "\n"
			<< "z_end_m=" << pc.z_end << "\n"
			<< "angle_list_deg=" << AngleListToString(config.angle_list_deg) << "\n"
			<< "tx_delay_offset_samples=" << pc.tx_delay_offset_samples << "\n"
			<< "tx_reference_mode=" << config.tx_reference_mode << "\n"
			<< "bandpass_enabled=" << (config.bandpass_enabled ? "true" : "false") << "\n"
			<< "bandpass_filter_path=" << (config.bandpass_filter_path.empty() ? std::string("") : config.bandpass_filter_path.string()) << "\n"
			<< "bandpass_taps=" << config.bandpass_taps << "\n"
			<< "motion_correction_enabled=" << (config.motion_enabled ? "true" : "false") << "\n"
			<< "motion_search_radius_px=[" << config.motion_search_radius_x << "," << config.motion_search_radius_z << "]\n"
			<< "motion_roi_m=[" << config.motion_roi_x_start_m << "," << config.motion_roi_x_end_m
			<< "," << config.motion_roi_z_start_m << "," << config.motion_roi_z_end_m << "]\n"
			<< "motion_stride_px=[" << config.motion_stride_x << "," << config.motion_stride_z << "]\n"
			<< "motion_smooth_radius=" << config.motion_smooth_radius << "\n"
			<< "motion_shift_csv=" << (motion_csv_path.empty() ? std::string("") : motion_csv_path.string()) << "\n"
			<< "motion_min_dx_px=" << motion_summary.min_dx << "\n"
			<< "motion_max_dx_px=" << motion_summary.max_dx << "\n"
			<< "motion_min_dz_px=" << motion_summary.min_dz << "\n"
			<< "motion_max_dz_px=" << motion_summary.max_dz << "\n"
			<< "motion_mean_abs_dx_px=" << motion_summary.mean_abs_dx << "\n"
			<< "motion_mean_abs_dz_px=" << motion_summary.mean_abs_dz << "\n"
			<< "motion_elapsed_s=" << motion_summary.elapsed_s << "\n"
			<< "block_svd_enabled=" << (block_svd_summary.enabled ? "true" : "false") << "\n"
			<< "block_svd_size_px=[" << block_svd_summary.block_width << "," << block_svd_summary.block_height << "]\n"
			<< "block_svd_block_count=" << block_svd_summary.block_count << "\n"
			<< "svd_auto_ssm=" << (config.svd_auto_ssm ? "true" : "false") << "\n"
			<< "ssm_max_components=" << ssm_result.max_components << "\n"
			<< "ssm_tissue_component_count=" << ssm_result.tissue_component_count << "\n"
			<< "ssm_flow_component_end=" << ssm_result.flow_component_end << "\n"
			<< "ssm_recommended_cutoff=[" << ssm_result.cutoff_start << "," << ssm_result.cutoff_end << "]\n"
			<< "ssm_fit_score=" << ssm_result.fit_score << "\n"
			<< "ssm_elapsed_ms=" << ssm_result.elapsed_ms << "\n"
			<< "svd_cutoff_one_based=[" << cutoff_low_rank << "," << cutoff_high_rank << "]\n"
			<< "svd_filter=temporal_covariance_eigendecomposition_equivalent_to_right_singular_vector_projection\n"
			<< "removed_largest_singular_components=1.." << (cutoff_low_rank - 1) << "\n"
			<< "kept_singular_components=" << cutoff_low_rank << ".." << cutoff_high_rank << "\n"
			<< "removed_smallest_singular_components=" << (cutoff_high_rank + 1) << ".." << frame_count << "\n"
			<< "power_sum_bin=" << power_path.string() << "\n"
			<< "power_db_bin=" << db_path.string() << "\n"
			<< "power_min=" << *power_minmax.first << "\n"
			<< "power_max=" << *power_minmax.second << "\n"
			<< "db_min=" << *db_minmax.first << "\n"
			<< "db_max=" << *db_minmax.second << "\n"
			<< "db_mean=" << db_mean << "\n"
			<< "db_std=" << db_std << "\n"
			<< "display_range_db=[" << -display_dr << ",0]\n"
			<< "display_config_db=[" << config.display_db_min << "," << config.display_db_max << "]\n"
			<< "read_s=" << read_s << "\n"
			<< "bandpass_s=" << bandpass_s << "\n"
			<< "rf_to_iq_s=" << rf_to_iq_s << "\n"
			<< "beamform_s=" << beamform_s << "\n"
			<< "covariance_s=" << covariance_s << "\n"
			<< "eigendecomposition_s=" << eig_s << "\n"
			<< "projection_build_s=" << projection_s << "\n"
			<< "svd_projection_apply_s=" << filter_s << "\n"
			<< "block_svd_gather_s=" << block_svd_summary.gather_s << "\n"
			<< "block_svd_covariance_s=" << block_svd_summary.covariance_s << "\n"
			<< "block_svd_eigendecomposition_s=" << block_svd_summary.eig_s << "\n"
			<< "block_svd_projection_build_s=" << block_svd_summary.projection_s << "\n"
			<< "block_svd_projection_apply_s=" << block_svd_summary.filter_s << "\n"
			<< "block_svd_scatter_s=" << block_svd_summary.scatter_s << "\n";

		if (!eigenvalues.empty())
		{
			out << "smallest_covariance_eigenvalue=" << eigenvalues.front() << "\n"
				<< "largest_covariance_eigenvalue=" << eigenvalues.back() << "\n";
		}
	}
}

int main(int argc, char** argv)
{
	float2* d_iq_cache = nullptr;
	float2* d_filtered = nullptr;
	cuComplex* d_projection = nullptr;
	cublasHandle_t cublas = nullptr;
	cusolverDnHandle_t solver = nullptr;

	try
	{
		const DopplerRunConfig config = LoadConfigFromArgs(argc, argv);
		ConfigureJindongmai(config);
		const int frame_count = ValidateInputFileAndGetFrames(config.input_path, config);
		std::filesystem::create_directories(config.output_dir);

		int device = 0;
		cudaDeviceProp prop{};
		CUDA_CHECK(cudaGetDevice(&device));
		CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
		std::cout << "[Info] Jindongmai Doppler on " << prop.name << "\n"
			<< "[Info] config=" << (config.config_path.empty() ? std::string("cli") : config.config_path.string()) << "\n"
			<< "[Info] input=" << config.input_path.string() << "\n"
			<< "[Info] frames=" << frame_count
			<< ", file_layout=[frame][seq][sample][channel] int16 RF"
			<< ", cuda_layout=[frame][seq][channel][sample]\n"
			<< "[Info] channel_order=" << config.channel_order
			<< " -> internal=ascending_x"
			<< ", tx_reference_mode=" << config.tx_reference_mode << "\n";

		const int pixels = pc.img_nx * pc.img_nz;
		const size_t iq_cache_elements = static_cast<size_t>(pixels) * frame_count;
		const size_t iq_cache_bytes = iq_cache_elements * sizeof(float2);

		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_iq_cache), iq_cache_bytes));
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_filtered), iq_cache_bytes));
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_projection), static_cast<size_t>(frame_count) * frame_count * sizeof(cuComplex)));
		CUBLAS_CHECK(cublasCreate(&cublas));
		CUSOLVER_CHECK(cusolverDnCreate(&solver));

		double read_s = 0.0;
		double bandpass_s = 0.0;
		double rf_to_iq_s = 0.0;
		double beamform_s = 0.0;
		BeamformAllFramesChunked(
			config.input_path,
			frame_count,
			d_iq_cache,
			read_s,
			bandpass_s,
			rf_to_iq_s,
			beamform_s,
			config);

		MotionCorrectionSummary motion_summary{};
		std::filesystem::path motion_csv_path;
		if (config.motion_enabled)
		{
			const MotionCorrectionConfig motion_config = BuildMotionCorrectionConfig(config);
			std::cout << "[Motion] enabled, roi_px=["
				<< motion_config.roi_x0 << "," << motion_config.roi_x1 << ","
				<< motion_config.roi_z0 << "," << motion_config.roi_z1 << "], search_px=["
				<< motion_config.search_radius_x << "," << motion_config.search_radius_z << "]\n";
			std::vector<float2> h_iq_cache(iq_cache_elements);
			CUDA_CHECK(cudaMemcpy(h_iq_cache.data(), d_iq_cache, iq_cache_bytes, cudaMemcpyDeviceToHost));
			motion_summary = ApplyGlobalMotionCorrection(
				h_iq_cache,
				config.img_nx,
				config.img_nz,
				frame_count,
				motion_config);
			CUDA_CHECK(cudaMemcpy(d_iq_cache, h_iq_cache.data(), iq_cache_bytes, cudaMemcpyHostToDevice));
			motion_csv_path = config.output_dir / "motion_shifts.csv";
			WriteMotionShiftCsv(motion_csv_path, motion_summary);
			std::cout << "[Motion] dx_range=[" << motion_summary.min_dx << "," << motion_summary.max_dx
				<< "], dz_range=[" << motion_summary.min_dz << "," << motion_summary.max_dz
				<< "], mean_abs=[" << motion_summary.mean_abs_dx << "," << motion_summary.mean_abs_dz
				<< "], elapsed=" << motion_summary.elapsed_s << " s\n";
		}

		SsmCutoffConfig ssm_config{};
		ssm_config.max_components = config.ssm_max_components > 0 ? config.ssm_max_components : frame_count;
		ssm_config.min_tissue = config.ssm_min_tissue;
		ssm_config.min_flow = config.ssm_min_flow;
		SsmCutoffResult ssm_result{};
		EstimateSsmCutoffFromIqCache(
			d_iq_cache,
			pixels,
			frame_count,
			ssm_config,
			&ssm_result);
		std::cout << "[SSM] recommended_cutoff=[" << ssm_result.cutoff_start
			<< "," << ssm_result.cutoff_end << "]"
			<< ", tissue_components=" << ssm_result.tissue_component_count
			<< ", flow_end=" << ssm_result.flow_component_end
			<< ", fit_score=" << ssm_result.fit_score << "\n";

		const int cutoff_low_rank = config.svd_auto_ssm ? ssm_result.cutoff_start : config.cutoff_low;
		const int cutoff_high_rank = std::min(config.svd_auto_ssm ? ssm_result.cutoff_end : config.cutoff_high, frame_count);
		if (cutoff_high_rank < cutoff_low_rank)
		{
			throw std::runtime_error("Frame count is smaller than cutoff low rank.");
		}
		std::cout << "[Info] using_svd_cutoff=[" << cutoff_low_rank << "," << cutoff_high_rank << "]\n";

		std::vector<float> eigenvalues;
		double covariance_s = 0.0;
		double eig_s = 0.0;
		double projection_s = 0.0;
		double filter_s = 0.0;
		BlockSvdSummary block_svd_summary{};
		if (config.block_svd_enabled)
		{
			std::cout << "[BlockSVD] enabled, block_size=["
				<< config.block_svd_width << "," << config.block_svd_height << "]\n";
			block_svd_summary = RunBlockwiseSvdFilter(
				cublas,
				solver,
				d_iq_cache,
				d_filtered,
				d_projection,
				eigenvalues,
				frame_count,
				cutoff_low_rank,
				cutoff_high_rank,
				config);
			covariance_s = block_svd_summary.covariance_s;
			eig_s = block_svd_summary.eig_s;
			projection_s = block_svd_summary.projection_s;
			filter_s = block_svd_summary.filter_s;
		}
		else
		{
			BuildSvdRankProjection(
				cublas,
				solver,
				d_iq_cache,
				d_projection,
				eigenvalues,
				frame_count,
				cutoff_low_rank,
				cutoff_high_rank,
				covariance_s,
				eig_s,
				projection_s);

			ApplyProjection(cublas, d_iq_cache, d_projection, d_filtered, frame_count, filter_s);
		}

		std::vector<float2> h_filtered(iq_cache_elements);
		CUDA_CHECK(cudaMemcpy(h_filtered.data(), d_filtered, iq_cache_bytes, cudaMemcpyDeviceToHost));

		std::vector<float> power;
		std::vector<float> db;
		double db_mean = 0.0;
		double db_std = 0.0;
		double display_dr = 0.0;
		ComputePowerAndDb(h_filtered, frame_count, power, db, db_mean, db_std, display_dr);

		std::ostringstream stem;
		stem << config.input_path.stem().string() << "_svd"
			<< std::setw(3) << std::setfill('0') << cutoff_low_rank
			<< "to" << std::setw(3) << std::setfill('0') << cutoff_high_rank;
		if (config.block_svd_enabled)
		{
			stem << "_blocksvd"
				<< std::setw(3) << std::setfill('0') << config.block_svd_width
				<< "x" << std::setw(3) << std::setfill('0') << config.block_svd_height;
		}
		const std::filesystem::path power_path = config.output_dir / (stem.str() + "_power_sum.bin");
		const std::filesystem::path db_path = config.output_dir / (stem.str() + "_power_db.bin");
		const std::filesystem::path summary_path = config.output_dir / (stem.str() + "_summary.txt");
		WriteFloatBin(power_path, power);
		WriteFloatBin(db_path, db);
		WriteSummary(
			summary_path,
			config,
			config.input_path,
			power_path,
			db_path,
			power,
			db,
			eigenvalues,
			ssm_result,
			motion_summary,
			motion_csv_path,
			block_svd_summary,
			frame_count,
			cutoff_low_rank,
			cutoff_high_rank,
			read_s,
			bandpass_s,
			rf_to_iq_s,
			beamform_s,
			covariance_s,
			eig_s,
			projection_s,
			filter_s,
			db_mean,
			db_std,
			display_dr);

		std::cout << std::fixed << std::setprecision(3)
			<< "[Timing] read=" << read_s << " s, bandpass=" << bandpass_s
			<< " s, rf_to_iq=" << rf_to_iq_s
			<< " s, beamform=" << beamform_s << " s\n"
			<< "[Timing] covariance=" << covariance_s << " s, eig=" << eig_s
			<< " s, projection=" << projection_s << " s, filter=" << filter_s << " s\n"
			<< "[Timing] block_gather=" << block_svd_summary.gather_s
			<< " s, block_scatter=" << block_svd_summary.scatter_s << " s\n"
			<< "[Output] power=" << power_path.string() << "\n"
			<< "[Output] db=" << db_path.string() << "\n"
			<< "[Output] summary=" << summary_path.string() << "\n"
			<< "[Display] adaptive_range_db=[" << -display_dr << ",0]\n"
			<< "[Display] configured_range_db=[" << config.display_db_min << "," << config.display_db_max << "]\n";

		CUSOLVER_CHECK(cusolverDnDestroy(solver));
		solver = nullptr;
		CUBLAS_CHECK(cublasDestroy(cublas));
		cublas = nullptr;
		CUDA_CHECK(cudaFree(d_projection));
		d_projection = nullptr;
		CUDA_CHECK(cudaFree(d_filtered));
		d_filtered = nullptr;
		CUDA_CHECK(cudaFree(d_iq_cache));
		d_iq_cache = nullptr;
	}
	catch (const std::exception& ex)
	{
		if (solver != nullptr) { cusolverDnDestroy(solver); }
		if (cublas != nullptr) { cublasDestroy(cublas); }
		if (d_projection != nullptr) { cudaFree(d_projection); }
		if (d_filtered != nullptr) { cudaFree(d_filtered); }
		if (d_iq_cache != nullptr) { cudaFree(d_iq_cache); }
		std::cerr << "[Error] " << ex.what() << "\n";
		return 1;
	}

	return 0;
}
