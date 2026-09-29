#include "beamformer.cuh"
#include "config.cuh"
#include "PARAMs.h"

#include <algorithm>
#include <chrono>
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
	struct TimingStats
	{
		float avg_ms = 0.0f;
		float median_ms = 0.0f;
		float min_ms = 0.0f;
		float max_ms = 0.0f;
	};

	std::vector<int> ParseFrameCountsCsv(const std::string& csv)
	{
		std::vector<int> values;
		std::stringstream ss(csv);
		std::string item;
		while (std::getline(ss, item, ','))
		{
			if (!item.empty())
			{
				values.push_back(std::stoi(item));
			}
		}
		if (values.empty())
		{
			throw std::runtime_error("Frame count list is empty.");
		}
		return values;
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

	void ConfigureZenodo7883227PowerDoppler()
	{
		constexpr float kPi = 3.14159265358979323846f;
		constexpr float kWavelength = 0.09856e-3f;
		constexpr float kOriginXWavelength = -64.427759740259731f;
		constexpr float kDeltaXWavelength = 0.50730519480519476f;
		constexpr float kOriginZWavelength = 18.0f;
		constexpr float kDeltaZWavelength = 0.5f;
		constexpr float kStartDepthWavelength = 10.0f;
		constexpr float kTransmitWavePeakCycles = 2.84375f;

		pc.seq = 5;
		pc.datalen = 256;
		pc.channels = 128;
		pc.f0 = 15.625e6f;
		pc.fs = 15.625e6f;
		pc.sound_speed = 1540.0f;
		pc.pitch = 0.1e-3f;
		pc.img_nx = 256;
		pc.img_nz = 180;
		pc.x_start = kOriginXWavelength * kWavelength;
		pc.x_end = (kOriginXWavelength + (pc.img_nx - 1) * kDeltaXWavelength) * kWavelength;
		pc.z_start = kOriginZWavelength * kWavelength;
		pc.z_end = (kOriginZWavelength + (pc.img_nz - 1) * kDeltaZWavelength) * kWavelength;
		pc.F_number = 1.9f;
		pc.minimum_aperture = 1e-3f;
		pc.show_range = 40;
		pc.use_auto_angle_span = false;
		pc.init_angle = -6.0f * (kPi / 180.0f);
		pc.final_angle = 6.0f * (kPi / 180.0f);

		const float t0 =
			2.0f * kStartDepthWavelength * kWavelength / pc.sound_speed -
			kTransmitWavePeakCycles / pc.f0;
		pc.tx_delay_offset_samples = t0 * pc.fs;
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

	std::vector<float2> PackLeadingFramesSubsetToHost(
		const std::vector<float2>& host_full,
		int nseq,
		int nch,
		int nsamples,
		int total_frames,
		int subset_frames)
	{
		if (subset_frames <= 0 || subset_frames > total_frames)
		{
			throw std::runtime_error("subset_frames must stay within 1..total_frames.");
		}

		const size_t blocks = static_cast<size_t>(nseq) * nch * nsamples;
		std::vector<float2> packed(static_cast<size_t>(subset_frames) * blocks);
		const size_t copy_bytes = static_cast<size_t>(subset_frames) * sizeof(float2);
		for (size_t block_idx = 0; block_idx < blocks; ++block_idx)
		{
			const float2* src = host_full.data() + block_idx * static_cast<size_t>(total_frames);
			float2* dst = packed.data() + block_idx * static_cast<size_t>(subset_frames);
			std::memcpy(dst, src, copy_bytes);
		}
		return packed;
	}

	int RunBeamformingBenchmark(
		const std::filesystem::path& input_path,
		const std::filesystem::path& output_csv,
		const std::vector<int>& frame_counts,
		int warmup_repeats,
		int benchmark_repeats)
	{
		ConfigureZenodo7883227PowerDoppler();

		const int total_frames = 800;
		const int pixels = pc.img_nx * pc.img_nz;
		const size_t full_elements = static_cast<size_t>(pc.seq) * pc.channels * pc.datalen * total_frames;
		if (!std::filesystem::exists(input_path))
		{
			throw std::runtime_error("Missing input bin: " + input_path.string());
		}
		std::filesystem::create_directories(output_csv.parent_path());

		int device_id = 0;
		cudaDeviceProp prop{};
		CUDA_CHECK(cudaGetDevice(&device_id));
		CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

		const auto read_start = std::chrono::high_resolution_clock::now();
		const std::vector<float2> h_iq_full = LoadComplexDataToHostFlat(input_path, full_elements);
		const auto read_stop = std::chrono::high_resolution_clock::now();
		const double read_s = std::chrono::duration<double>(read_stop - read_start).count();

		std::ofstream csv(output_csv, std::ios::out | std::ios::trunc);
		if (!csv.is_open())
		{
			throw std::runtime_error("Cannot create benchmark CSV: " + output_csv.string());
		}
		csv << "method,frames,warmup_repeats,benchmark_repeats,read_s,pack_s,upload_s,beamform_avg_ms,beamform_median_ms,beamform_min_ms,beamform_max_ms,beamform_median_s\n";

		std::cout << "[Info] Our CUDA beamforming benchmark on " << prop.name << std::endl;
		std::cout << "[Info] input=" << input_path.string() << ", read_once=" << read_s << " s" << std::endl;

		BEAMFORMER beamform;
		for (int n_frames : frame_counts)
		{
			if (n_frames <= 0 || n_frames > total_frames)
			{
				throw std::runtime_error("Frame count must stay within 1..800.");
			}

			const auto pack_start = std::chrono::high_resolution_clock::now();
			const std::vector<float2> h_iq_packed = PackLeadingFramesSubsetToHost(
				h_iq_full,
				pc.seq,
				pc.channels,
				pc.datalen,
				total_frames,
				n_frames);
			const auto pack_stop = std::chrono::high_resolution_clock::now();
			const double pack_s = std::chrono::duration<double>(pack_stop - pack_start).count();

			float2* d_iq = nullptr;
			float2* d_iq_cache = nullptr;
			cudaEvent_t start_event = nullptr;
			cudaEvent_t stop_event = nullptr;

			try
			{
				const size_t packed_bytes = h_iq_packed.size() * sizeof(float2);
				const size_t iq_cache_elements = static_cast<size_t>(pixels) * static_cast<size_t>(n_frames);
				const size_t iq_cache_bytes = sizeof(float2) * iq_cache_elements;

				CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_iq), packed_bytes));
				CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_iq_cache), iq_cache_bytes));

				const auto upload_start = std::chrono::high_resolution_clock::now();
				CUDA_CHECK(cudaMemcpy(d_iq, h_iq_packed.data(), packed_bytes, cudaMemcpyHostToDevice));
				CUDA_CHECK(cudaDeviceSynchronize());
				const auto upload_stop = std::chrono::high_resolution_clock::now();
				const double upload_s = std::chrono::duration<double>(upload_stop - upload_start).count();

				for (int i = 0; i < warmup_repeats; ++i)
				{
					beamform.BeamformIQCacheSharedRectChannelOuterCore(d_iq, n_frames, d_iq_cache);
				}
				CUDA_CHECK(cudaDeviceSynchronize());

				CUDA_CHECK(cudaEventCreate(&start_event));
				CUDA_CHECK(cudaEventCreate(&stop_event));

				std::vector<float> samples_ms;
				samples_ms.reserve(static_cast<size_t>(benchmark_repeats));
				for (int i = 0; i < benchmark_repeats; ++i)
				{
					CUDA_CHECK(cudaEventRecord(start_event));
					beamform.BeamformIQCacheSharedRectChannelOuterCore(d_iq, n_frames, d_iq_cache);
					CUDA_CHECK(cudaEventRecord(stop_event));
					CUDA_CHECK(cudaEventSynchronize(stop_event));

					float elapsed_ms = 0.0f;
					CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));
					samples_ms.push_back(elapsed_ms);
				}

				const TimingStats stats = BuildStats(std::move(samples_ms));
				csv << "our_cuda,"
					<< n_frames << ","
					<< warmup_repeats << ","
					<< benchmark_repeats << ","
					<< read_s << ","
					<< pack_s << ","
					<< upload_s << ","
					<< stats.avg_ms << ","
					<< stats.median_ms << ","
					<< stats.min_ms << ","
					<< stats.max_ms << ","
					<< (static_cast<double>(stats.median_ms) * 1.0e-3)
					<< "\n";

				std::cout << std::fixed << std::setprecision(3)
					<< "[Bench] frames=" << n_frames
					<< ", beamform_median=" << stats.median_ms << " ms"
					<< ", fps=" << 1000.0f / std::max(stats.median_ms, 1.0e-6f)
					<< std::endl;

				CUDA_CHECK(cudaEventDestroy(stop_event));
				stop_event = nullptr;
				CUDA_CHECK(cudaEventDestroy(start_event));
				start_event = nullptr;
				CUDA_CHECK(cudaFree(d_iq_cache));
				d_iq_cache = nullptr;
				CUDA_CHECK(cudaFree(d_iq));
				d_iq = nullptr;
			}
			catch (...)
			{
				if (stop_event != nullptr) { cudaEventDestroy(stop_event); }
				if (start_event != nullptr) { cudaEventDestroy(start_event); }
				if (d_iq_cache != nullptr) { cudaFree(d_iq_cache); }
				if (d_iq != nullptr) { cudaFree(d_iq); }
				throw;
			}
		}

		std::cout << "[Info] Saved our CUDA beamforming CSV: " << output_csv.string() << std::endl;
		return 0;
	}

	void PrintUsage()
	{
		std::cerr << "Usage: OurCudaBeamformBench.exe <rf001_bin> <output_csv> [frame_counts_csv] [warmup] [repeat]\n";
	}
}

int main(int argc, char** argv)
{
	try
	{
		const std::filesystem::path input_path = (argc >= 2)
			? std::filesystem::path(argv[1])
			: std::filesystem::path("E:/Code/Cuda/LinearArrayCPWC_OPT_test/bins_7883227/RF_001_7883227_bs100bw_iq_float2.bin");
		const std::filesystem::path output_csv = (argc >= 3)
			? std::filesystem::path(argv[2])
			: std::filesystem::path("outputs/benchmark_rf001_our_cuda_beamforming.csv");
		const std::vector<int> frame_counts = (argc >= 4)
			? ParseFrameCountsCsv(argv[3])
			: std::vector<int>{64, 128, 256, 512, 800};
		const int warmup = (argc >= 5) ? std::stoi(argv[4]) : 1;
		const int repeat = (argc >= 6) ? std::stoi(argv[5]) : 5;
		return RunBeamformingBenchmark(input_path, output_csv, frame_counts, warmup, repeat);
	}
	catch (const std::exception& ex)
	{
		PrintUsage();
		std::cerr << "[Error] " << ex.what() << std::endl;
		return 1;
	}
}
