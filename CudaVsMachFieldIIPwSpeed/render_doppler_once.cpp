#include "beamformer.cuh"
#include "config.cuh"
#include "data_preprocess_gpu_v2.cuh"
#include "PARAMs.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
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

	void ConfigureFieldIIVerticalPw(int pw)
	{
		pc.seq = pw;
		pc.datalen = kFieldIISamples;
		pc.channels = kFieldIIChannels;
		pc.f0 = 0.0f;
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
}

int main(int argc, char** argv)
{
	try
	{
		const std::filesystem::path data_root =
			(argc >= 2) ? std::filesystem::path(argv[1]) : std::filesystem::path("E:/Code/mat/y_branch_fieldii_gpu_pd");
		const std::filesystem::path output_dir =
			(argc >= 3) ? std::filesystem::path(argv[2]) : std::filesystem::path("outputs");
		const int pw = (argc >= 4) ? std::stoi(argv[3]) : 5;
		if (pw < 1 || pw > 16)
		{
			throw std::runtime_error("pw must be within 1..16.");
		}

		std::filesystem::create_directories(output_dir);
		ConfigureFieldIIVerticalPw(pw);

		int device = 0;
		cudaDeviceProp prop{};
		CUDA_CHECK(cudaGetDevice(&device));
		CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
		std::cout << "[Info] Render Power Doppler on " << prop.name << "\n";

		const int pixels = pc.img_nx * pc.img_nz;
		const size_t full_elements = static_cast<size_t>(pc.seq) * pc.channels * pc.datalen * kFieldIITotalFrames;
		const size_t input_bytes = full_elements * sizeof(float);
		const std::filesystem::path input_path = ResolveFieldIIBin(data_root, pw);
		if (!std::filesystem::exists(input_path))
		{
			throw std::runtime_error("Missing input bin: " + input_path.string());
		}
		std::cout << "[Info] input_layout=[frame][seq][channel][sample]\n";

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
			const size_t iq_bytes = full_elements * sizeof(float2);
			const size_t iq_cache_elements = static_cast<size_t>(pixels) * kFieldIITotalFrames;
			const size_t iq_cache_bytes = iq_cache_elements * sizeof(float2);

			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rf), input_bytes));
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_iq), iq_bytes));
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_iq_cache), iq_cache_bytes));
			CUDA_CHECK(cudaMemcpy(d_rf, h_rf_full.data(), input_bytes, cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaDeviceSynchronize());

			FieldIIRfToIqPreprocessor rf_to_iq(
				pc.datalen,
				pc.channels,
				pc.seq,
				kFieldIITotalFrames);
			BEAMFORMER beamform;

			for (int i = 0; i < 3; ++i)
			{
				rf_to_iq.RfToIq(d_rf, d_iq);
				beamform.BeamformIQCacheSharedRectChannelOuterCore(d_iq, kFieldIITotalFrames, d_iq_cache);
			}
			CUDA_CHECK(cudaDeviceSynchronize());

			CUDA_CHECK(cudaEventCreate(&start_event));
			CUDA_CHECK(cudaEventCreate(&preprocess_event));
			CUDA_CHECK(cudaEventCreate(&stop_event));
			CUDA_CHECK(cudaEventRecord(start_event));
			rf_to_iq.RfToIq(d_rf, d_iq);
			CUDA_CHECK(cudaEventRecord(preprocess_event));
			beamform.BeamformIQCacheSharedRectChannelOuterCore(d_iq, kFieldIITotalFrames, d_iq_cache);
			CUDA_CHECK(cudaEventRecord(stop_event));
			CUDA_CHECK(cudaEventSynchronize(stop_event));

			float preprocess_ms = 0.0f;
			float total_ms = 0.0f;
			CUDA_CHECK(cudaEventElapsedTime(&preprocess_ms, start_event, preprocess_event));
			CUDA_CHECK(cudaEventElapsedTime(&total_ms, start_event, stop_event));

			std::vector<float2> h_iq_cache(iq_cache_elements);
			CUDA_CHECK(cudaMemcpy(h_iq_cache.data(), d_iq_cache, iq_cache_bytes, cudaMemcpyDeviceToHost));

			std::vector<float> power(static_cast<size_t>(pixels), 0.0f);
			for (int frame = 0; frame < kFieldIITotalFrames; ++frame)
			{
				const size_t frame_offset = static_cast<size_t>(frame) * pixels;
				for (int pixel = 0; pixel < pixels; ++pixel)
				{
					const float2 value = h_iq_cache[frame_offset + pixel];
					power[pixel] += value.x * value.x + value.y * value.y;
				}
			}
			for (float& value : power)
			{
				value /= static_cast<float>(kFieldIITotalFrames);
			}

			const auto minmax = std::minmax_element(power.begin(), power.end());
			const double sum_power = std::accumulate(power.begin(), power.end(), 0.0);
			const std::filesystem::path prefix = output_dir / ("fieldii_pw" + FormatPw(pw) + "_cuda_power_doppler");
			const std::filesystem::path power_bin = prefix.string() + ".bin";
			const std::filesystem::path meta_txt = prefix.string() + ".txt";
			WriteFloatBin(power_bin, power);

			std::ofstream meta(meta_txt);
			meta << "pw=" << pw << "\n"
				<< "frames=" << kFieldIITotalFrames << "\n"
				<< "channels=" << pc.channels << "\n"
				<< "samples=" << pc.datalen << "\n"
				<< "nx=" << pc.img_nx << "\n"
				<< "nz=" << pc.img_nz << "\n"
				<< "x_start_mm=" << pc.x_start * 1.0e3f << "\n"
				<< "x_end_mm=" << pc.x_end * 1.0e3f << "\n"
				<< "z_start_mm=" << pc.z_start * 1.0e3f << "\n"
				<< "z_end_mm=" << pc.z_end * 1.0e3f << "\n"
				<< "phase_mode=off\n"
				<< "read_s=" << read_s << "\n"
				<< "rf_to_iq_ms=" << preprocess_ms << "\n"
				<< "beamform_ms=" << total_ms - preprocess_ms << "\n"
				<< "total_cuda_ms=" << total_ms << "\n"
				<< "power_min=" << *minmax.first << "\n"
				<< "power_max=" << *minmax.second << "\n"
				<< "power_mean=" << sum_power / static_cast<double>(power.size()) << "\n";

			std::cout << std::fixed << std::setprecision(3)
				<< "[Render] pw=" << pw
				<< ", rf_to_iq=" << preprocess_ms << " ms"
				<< ", beamform=" << (total_ms - preprocess_ms) << " ms"
				<< ", total_cuda=" << total_ms << " ms\n"
				<< "[Render] power_bin=" << power_bin.string() << "\n"
				<< "[Render] meta=" << meta_txt.string() << "\n";

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
	catch (const std::exception& e)
	{
		std::cerr << "[Error] " << e.what() << "\n";
		return 1;
	}
	return 0;
}
