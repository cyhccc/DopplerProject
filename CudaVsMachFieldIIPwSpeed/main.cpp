#include "beamformer.cuh"
#include "config.cuh"
#include "data_preprocess_gpu_v2.cuh"
#include "PARAMs.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

PARAMs pc;

namespace
{
	constexpr int kFieldIITotalFrames = 128;
	constexpr int kFieldIIChannels = 128;
	constexpr int kFieldIISamples = 788;
	constexpr int kFieldIINx = 512;
	constexpr int kFieldIINz = 180;
	constexpr float kFieldIIPi = 3.14159265358979323846f;
	constexpr float kFieldIIFsHz = 30.0e6f;
	constexpr float kFieldIIF0Hz = 5.0e6f;
	constexpr float kFieldIISoundSpeed = 1540.0f;
	constexpr float kFieldIIPitchM = 0.09e-3f;
	constexpr float kFieldIIXStartM = -3.0e-3f;
	constexpr float kFieldIIXEndM = 3.0e-3f;
	constexpr float kFieldIIZStartM = 2.0e-3f;
	constexpr float kFieldIIZEndM = 10.0e-3f;
	constexpr float kFieldIIFNumber = 1.75f;
	constexpr float kFieldIIMinimumApertureM = 1.0e-3f;
	constexpr float kFieldIIInitAngleDeg = -9.0f;
	constexpr float kFieldIIFinalAngleDeg = 9.0f;
	constexpr float kFieldIIReferenceTStartSamples = 57.58396810051636f;

	struct TimingStats
	{
		float avg_ms = 0.0f;
		float median_ms = 0.0f;
		float min_ms = 0.0f;
		float max_ms = 0.0f;
	};

	std::vector<int> ParsePwValues(const std::string& text)
	{
		std::vector<int> values;
		const size_t colon = text.find(':');
		if (colon != std::string::npos)
		{
			const int first = std::stoi(text.substr(0, colon));
			const int last = std::stoi(text.substr(colon + 1));
			if (first > last)
			{
				throw std::runtime_error("PW range must be ascending.");
			}
			for (int pw = first; pw <= last; ++pw)
			{
				values.push_back(pw);
			}
		}
		else
		{
			std::stringstream ss(text);
			std::string item;
			while (std::getline(ss, item, ','))
			{
				if (!item.empty())
				{
					values.push_back(std::stoi(item));
				}
			}
		}

		if (values.empty())
		{
			throw std::runtime_error("PW value list is empty.");
		}
		for (int pw : values)
		{
			if (pw < 1 || pw > 16)
			{
				throw std::runtime_error("PW values must stay within 1..16 for the specialized CUDA path.");
			}
		}
		return values;
	}

	std::string ToLower(std::string value)
	{
		std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
			return static_cast<char>(std::tolower(ch));
		});
		return value;
	}

	bool ParsePhaseMode(const std::string& text)
	{
		const std::string value = ToLower(text);
		if (value == "on" || value == "phase_on" || value == "phase-on" || value == "1" || value == "true")
		{
			return true;
		}
		if (value == "off" || value == "phase_off" || value == "phase-off" || value == "0" || value == "false")
		{
			return false;
		}
		throw std::runtime_error("phase_mode must be on or off.");
	}

	TimingStats BuildStats(std::vector<float> samples_ms)
	{
		if (samples_ms.empty())
		{
			throw std::runtime_error("Timing sample list is empty.");
		}

		TimingStats stats{};
		for (float value : samples_ms)
		{
			stats.avg_ms += value;
		}
		stats.avg_ms /= static_cast<float>(samples_ms.size());
		stats.min_ms = *std::min_element(samples_ms.begin(), samples_ms.end());
		stats.max_ms = *std::max_element(samples_ms.begin(), samples_ms.end());

		std::sort(samples_ms.begin(), samples_ms.end());
		const size_t mid = samples_ms.size() / 2;
		stats.median_ms = (samples_ms.size() % 2 == 0)
			? 0.5f * (samples_ms[mid - 1] + samples_ms[mid])
			: samples_ms[mid];
		return stats;
	}

	std::string FormatPw(int pw)
	{
		std::ostringstream ss;
		ss << std::setw(2) << std::setfill('0') << pw;
		return ss.str();
	}

	std::filesystem::path ResolveFieldIIBin(const std::filesystem::path& data_root, int pw)
	{
		const std::string pw_text = FormatPw(pw);
		return data_root /
			("results_vertical_pw" + pw_text + "_128f") /
			"cuda_rf_frame_major_bin" /
			("vertical_pw" + pw_text + "_128f_rf_frame_seq_channel_sample_float.bin");
	}

	void ConfigureFieldIIVerticalPw(int pw, bool phase_on)
	{
		pc.seq = pw;
		pc.datalen = kFieldIISamples;
		pc.channels = kFieldIIChannels;
		pc.f0 = phase_on ? kFieldIIF0Hz : 0.0f;
		pc.fs = kFieldIIFsHz;
		pc.sound_speed = kFieldIISoundSpeed;
		pc.pitch = kFieldIIPitchM;
		pc.img_nx = kFieldIINx;
		pc.img_nz = kFieldIINz;
		pc.x_start = kFieldIIXStartM;
		pc.x_end = kFieldIIXEndM;
		pc.z_start = kFieldIIZStartM;
		pc.z_end = kFieldIIZEndM;
		pc.depth = kFieldIIZEndM;
		pc.F_number = kFieldIIFNumber;
		pc.minimum_aperture = kFieldIIMinimumApertureM;
		pc.show_range = 60;
		pc.use_auto_angle_span = false;
		pc.init_angle = kFieldIIInitAngleDeg * (kFieldIIPi / 180.0f);
		pc.final_angle = kFieldIIFinalAngleDeg * (kFieldIIPi / 180.0f);
		pc.tx_delay_offset_samples = kFieldIIReferenceTStartSamples;
		pc.UpdateParamter();
	}

	std::vector<float2> LoadComplexDataToHostFlat(
		const std::filesystem::path& filename,
		size_t total_elements)
	{
		const size_t file_size_bytes = total_elements * sizeof(float2);
		std::ifstream file(filename, std::ios::binary | std::ios::ate);
		if (!file.is_open())
		{
			throw std::runtime_error("Cannot open input file: " + filename.string());
		}

		const size_t actual_size = static_cast<size_t>(file.tellg());
		if (actual_size != file_size_bytes)
		{
			throw std::runtime_error(
				"Unexpected file size for " + filename.string() +
				". Expected " + std::to_string(file_size_bytes) +
				" bytes, got " + std::to_string(actual_size) + " bytes.");
		}

		file.seekg(0, std::ios::beg);
		std::vector<float2> h_buffer(total_elements);
		file.read(reinterpret_cast<char*>(h_buffer.data()), static_cast<std::streamsize>(file_size_bytes));
		if (!file)
		{
			throw std::runtime_error("File read incomplete for " + filename.string());
		}
		return h_buffer;
	}

	std::vector<float> LoadFloatDataToHostFlat(
		const std::filesystem::path& filename,
		size_t total_elements)
	{
		const size_t file_size_bytes = total_elements * sizeof(float);
		std::ifstream file(filename, std::ios::binary | std::ios::ate);
		if (!file.is_open())
		{
			throw std::runtime_error("Cannot open input file: " + filename.string());
		}

		const size_t actual_size = static_cast<size_t>(file.tellg());
		if (actual_size != file_size_bytes)
		{
			throw std::runtime_error(
				"Unexpected file size for " + filename.string() +
				". Expected " + std::to_string(file_size_bytes) +
				" bytes, got " + std::to_string(actual_size) + " bytes.");
		}

		file.seekg(0, std::ios::beg);
		std::vector<float> h_buffer(total_elements);
		file.read(reinterpret_cast<char*>(h_buffer.data()), static_cast<std::streamsize>(file_size_bytes));
		if (!file)
		{
			throw std::runtime_error("File read incomplete for " + filename.string());
		}
		return h_buffer;
	}

	void WriteCsvHeader(std::ofstream& csv)
	{
		csv << "method,pw,frames,channels,samples,nx,nz,phase_mode,input_layout,input_bytes,input_gb,"
			<< "warmup_repeats,benchmark_repeats,read_s,upload_s,"
			<< "rf_to_iq_avg_ms,rf_to_iq_median_ms,rf_to_iq_min_ms,rf_to_iq_max_ms,"
			<< "beamform_avg_ms,beamform_median_ms,beamform_min_ms,beamform_max_ms,"
			<< "total_cuda_avg_ms,total_cuda_median_ms,total_cuda_min_ms,total_cuda_max_ms,"
			<< "total_cuda_median_s,total_cuda_median_ms_per_frame,"
			<< "beamform_median_s,beamform_median_ms_per_frame\n";
	}

	void BenchmarkOnePw(
		const std::filesystem::path& input_path,
		std::ofstream& csv,
		int pw,
		int warmup_repeats,
		int benchmark_repeats,
		bool phase_on)
	{
		ConfigureFieldIIVerticalPw(pw, phase_on);

		const int pixels = pc.img_nx * pc.img_nz;
		const size_t full_elements = static_cast<size_t>(pc.seq) * pc.channels * pc.datalen * kFieldIITotalFrames;
		const size_t input_bytes = full_elements * sizeof(float);
		if (!std::filesystem::exists(input_path))
		{
			throw std::runtime_error("Missing input bin: " + input_path.string());
		}

		const auto read_start = std::chrono::high_resolution_clock::now();
		const std::vector<float> h_rf_full = LoadFloatDataToHostFlat(input_path, full_elements);
		const auto read_stop = std::chrono::high_resolution_clock::now();
		const double read_s = std::chrono::duration<double>(read_stop - read_start).count();

		float* d_rf = nullptr;
		float2* d_iq = nullptr;
		float2* d_iq_cache = nullptr;
		cudaEvent_t start_event = nullptr;
		cudaEvent_t preprocess_event = nullptr;
		cudaEvent_t stop_event = nullptr;

		try
		{
			const size_t iq_elements = full_elements;
			const size_t iq_bytes = iq_elements * sizeof(float2);
			const size_t iq_cache_elements = static_cast<size_t>(pixels) * static_cast<size_t>(kFieldIITotalFrames);
			const size_t iq_cache_bytes = sizeof(float2) * iq_cache_elements;

			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rf), input_bytes));
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_iq), iq_bytes));
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_iq_cache), iq_cache_bytes));

			const auto upload_start = std::chrono::high_resolution_clock::now();
			CUDA_CHECK(cudaMemcpy(d_rf, h_rf_full.data(), input_bytes, cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaDeviceSynchronize());
			const auto upload_stop = std::chrono::high_resolution_clock::now();
			const double upload_s = std::chrono::duration<double>(upload_stop - upload_start).count();

			FieldIIRfToIqPreprocessor rf_to_iq(
				pc.datalen,
				pc.channels,
				pc.seq,
				kFieldIITotalFrames);
			BEAMFORMER beamform;
			for (int i = 0; i < warmup_repeats; ++i)
			{
				rf_to_iq.RfToIq(d_rf, d_iq);
				beamform.BeamformIQCacheSharedRectChannelOuterCore(
					d_iq,
					kFieldIITotalFrames,
					d_iq_cache);
			}
			CUDA_CHECK(cudaDeviceSynchronize());

			CUDA_CHECK(cudaEventCreate(&start_event));
			CUDA_CHECK(cudaEventCreate(&preprocess_event));
			CUDA_CHECK(cudaEventCreate(&stop_event));

			std::vector<float> preprocess_samples_ms;
			std::vector<float> beamform_samples_ms;
			std::vector<float> total_samples_ms;
			preprocess_samples_ms.reserve(static_cast<size_t>(benchmark_repeats));
			beamform_samples_ms.reserve(static_cast<size_t>(benchmark_repeats));
			total_samples_ms.reserve(static_cast<size_t>(benchmark_repeats));
			for (int i = 0; i < benchmark_repeats; ++i)
			{
				CUDA_CHECK(cudaEventRecord(start_event));
				rf_to_iq.RfToIq(d_rf, d_iq);
				CUDA_CHECK(cudaEventRecord(preprocess_event));
				beamform.BeamformIQCacheSharedRectChannelOuterCore(
					d_iq,
					kFieldIITotalFrames,
					d_iq_cache);
				CUDA_CHECK(cudaEventRecord(stop_event));
				CUDA_CHECK(cudaEventSynchronize(stop_event));

				float preprocess_ms = 0.0f;
				float total_ms = 0.0f;
				CUDA_CHECK(cudaEventElapsedTime(&preprocess_ms, start_event, preprocess_event));
				CUDA_CHECK(cudaEventElapsedTime(&total_ms, start_event, stop_event));
				preprocess_samples_ms.push_back(preprocess_ms);
				total_samples_ms.push_back(total_ms);
				beamform_samples_ms.push_back(total_ms - preprocess_ms);
			}

			const TimingStats preprocess_stats = BuildStats(std::move(preprocess_samples_ms));
			const TimingStats beamform_stats = BuildStats(std::move(beamform_samples_ms));
			const TimingStats total_stats = BuildStats(std::move(total_samples_ms));
			const std::string phase_mode = phase_on ? "on" : "off";
			const double input_gb = static_cast<double>(input_bytes) / (1024.0 * 1024.0 * 1024.0);
			const double total_median_s = static_cast<double>(total_stats.median_ms) * 1.0e-3;
			const double total_median_ms_per_frame = static_cast<double>(total_stats.median_ms) / kFieldIITotalFrames;
			const double beamform_median_s = static_cast<double>(beamform_stats.median_ms) * 1.0e-3;
			const double beamform_median_ms_per_frame = static_cast<double>(beamform_stats.median_ms) / kFieldIITotalFrames;

			csv << "our_cuda_rf_gpu_hilbert,"
				<< pw << ","
				<< kFieldIITotalFrames << ","
				<< pc.channels << ","
				<< pc.datalen << ","
				<< pc.img_nx << ","
				<< pc.img_nz << ","
				<< phase_mode << ","
				<< "[frame][seq][channel][sample],"
				<< input_bytes << ","
				<< input_gb << ","
				<< warmup_repeats << ","
				<< benchmark_repeats << ","
				<< read_s << ","
				<< upload_s << ","
				<< preprocess_stats.avg_ms << ","
				<< preprocess_stats.median_ms << ","
				<< preprocess_stats.min_ms << ","
				<< preprocess_stats.max_ms << ","
				<< beamform_stats.avg_ms << ","
				<< beamform_stats.median_ms << ","
				<< beamform_stats.min_ms << ","
				<< beamform_stats.max_ms << ","
				<< total_stats.avg_ms << ","
				<< total_stats.median_ms << ","
				<< total_stats.min_ms << ","
				<< total_stats.max_ms << ","
				<< total_median_s << ","
				<< total_median_ms_per_frame << ","
				<< beamform_median_s << ","
				<< beamform_median_ms_per_frame
				<< "\n";

			std::cout << std::fixed << std::setprecision(3)
				<< "[Bench] pw=" << pw
				<< ", input_layout=[frame][seq][channel][sample]"
				<< ", input=" << input_gb << " GiB"
				<< ", rf_to_iq_median=" << preprocess_stats.median_ms << " ms"
				<< ", beamform_median=" << beamform_stats.median_ms << " ms"
				<< ", total_cuda_median=" << total_stats.median_ms << " ms"
				<< ", total_ms/frame=" << total_median_ms_per_frame
				<< std::endl;

			CUDA_CHECK(cudaEventDestroy(stop_event));
			stop_event = nullptr;
			CUDA_CHECK(cudaEventDestroy(preprocess_event));
			preprocess_event = nullptr;
			CUDA_CHECK(cudaEventDestroy(start_event));
			start_event = nullptr;
			CUDA_CHECK(cudaFree(d_iq_cache));
			d_iq_cache = nullptr;
			CUDA_CHECK(cudaFree(d_iq));
			d_iq = nullptr;
			CUDA_CHECK(cudaFree(d_rf));
			d_rf = nullptr;
		}
		catch (...)
		{
			if (stop_event != nullptr) { cudaEventDestroy(stop_event); }
			if (preprocess_event != nullptr) { cudaEventDestroy(preprocess_event); }
			if (start_event != nullptr) { cudaEventDestroy(start_event); }
			if (d_iq_cache != nullptr) { cudaFree(d_iq_cache); }
			if (d_iq != nullptr) { cudaFree(d_iq); }
			if (d_rf != nullptr) { cudaFree(d_rf); }
			throw;
		}
	}

	int RunFieldIIPwBenchmark(
		const std::filesystem::path& data_root,
		const std::filesystem::path& output_csv,
		const std::vector<int>& pw_values,
		int warmup_repeats,
		int benchmark_repeats,
		bool phase_on)
	{
		if (!std::filesystem::exists(data_root))
		{
			throw std::runtime_error("Missing Field II data root: " + data_root.string());
		}
		if (output_csv.has_parent_path())
		{
			std::filesystem::create_directories(output_csv.parent_path());
		}

		int device_id = 0;
		cudaDeviceProp prop{};
		CUDA_CHECK(cudaGetDevice(&device_id));
		CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

		std::ofstream csv(output_csv, std::ios::out | std::ios::trunc);
		if (!csv.is_open())
		{
			throw std::runtime_error("Cannot create benchmark CSV: " + output_csv.string());
		}
		WriteCsvHeader(csv);

		std::cout << "[Info] Our CUDA Field II PW sweep on " << prop.name << std::endl;
		std::cout << "[Info] data_root=" << data_root.string() << std::endl;
		std::cout << "[Info] frames=" << kFieldIITotalFrames
			<< ", grid=" << kFieldIINx << "x" << kFieldIINz
			<< ", samples=" << kFieldIISamples
			<< ", phase_mode=" << (phase_on ? "on" : "off")
			<< std::endl;

		for (int pw : pw_values)
		{
			const std::filesystem::path input_path = ResolveFieldIIBin(data_root, pw);
			BenchmarkOnePw(input_path, csv, pw, warmup_repeats, benchmark_repeats, phase_on);
		}

		std::cout << "[Info] Saved our CUDA Field II PW CSV: " << output_csv.string() << std::endl;
		return 0;
	}

	void PrintUsage()
	{
		std::cerr
			<< "Usage: FieldIIPwCudaBeamformBench.exe <fieldii_data_root> <output_csv> "
			<< "[pw_values_csv_or_range] [warmup] [repeat] [phase_mode]\n"
			<< "Example: FieldIIPwCudaBeamformBench.exe "
			<< "\"E:/Code/mat/y_branch_fieldii_gpu_pd\" outputs/benchmark_fieldii_pw_our_cuda.csv "
			<< "3:16 1 5 on\n";
	}
}

int main(int argc, char** argv)
{
	try
	{
		const std::filesystem::path data_root = (argc >= 2)
			? std::filesystem::path(argv[1])
			: std::filesystem::path("E:/Code/mat/y_branch_fieldii_gpu_pd");
		const std::filesystem::path output_csv = (argc >= 3)
			? std::filesystem::path(argv[2])
			: std::filesystem::path("outputs/benchmark_fieldii_pw_our_cuda.csv");
		const std::vector<int> pw_values = (argc >= 4)
			? ParsePwValues(argv[3])
			: ParsePwValues("3:16");
		const int warmup = (argc >= 5) ? std::stoi(argv[4]) : 1;
		const int repeat = (argc >= 6) ? std::stoi(argv[5]) : 5;
		const bool phase_on = (argc >= 7) ? ParsePhaseMode(argv[6]) : true;

		return RunFieldIIPwBenchmark(data_root, output_csv, pw_values, warmup, repeat, phase_on);
	}
	catch (const std::exception& ex)
	{
		PrintUsage();
		std::cerr << "[Error] " << ex.what() << std::endl;
		return 1;
	}
}
