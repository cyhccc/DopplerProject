#include "beamformer.cuh"
#include "config.cuh"
#include "PARAMs.h"
#include "rsvd_filter.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
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

	void SaveFloatBinary(
		const std::filesystem::path& filename,
		const std::vector<float>& image)
	{
		std::filesystem::create_directories(filename.parent_path());

		std::ofstream file(filename, std::ios::binary);
		if (!file.is_open())
		{
			throw std::runtime_error("Cannot create float output file: " + filename.string());
		}

		file.write(reinterpret_cast<const char*>(image.data()), static_cast<std::streamsize>(image.size() * sizeof(float)));
		if (!file)
		{
			throw std::runtime_error("Failed while writing float output file: " + filename.string());
		}
	}

	std::vector<std::filesystem::path> ListBinFiles(
		const std::filesystem::path& bin_dir,
		int max_files)
	{
		if (!std::filesystem::exists(bin_dir))
		{
			throw std::runtime_error("Bin directory does not exist: " + bin_dir.string());
		}

		std::vector<std::filesystem::path> files;
		for (const auto& entry : std::filesystem::directory_iterator(bin_dir))
		{
			if (!entry.is_regular_file())
			{
				continue;
			}

			const std::filesystem::path path = entry.path();
			if (path.extension() == ".bin" && path.filename().string().find("RF_") == 0)
			{
				files.push_back(path);
			}
		}

		std::sort(files.begin(), files.end());
		if (files.empty())
		{
			throw std::runtime_error("No RF_*.bin files found under " + bin_dir.string());
		}

		if (max_files > 0 && static_cast<int>(files.size()) > max_files)
		{
			files.resize(static_cast<size_t>(max_files));
		}
		return files;
	}

	void ReadComplexDataToPinnedBuffer(
		const std::filesystem::path& filename,
		float2* h_buffer,
		size_t total_elements)
	{
		if (h_buffer == nullptr)
		{
			throw std::runtime_error("Pinned host buffer must not be null.");
		}

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
		file.read(reinterpret_cast<char*>(h_buffer), static_cast<std::streamsize>(file_size_bytes));
		if (!file)
		{
			throw std::runtime_error("File read incomplete for " + filename.string());
		}
	}

	void AccumulatePowerBlock(
		std::vector<double>& power_sum,
		const float* h_power,
		size_t pixels)
	{
		for (size_t i = 0; i < pixels; ++i)
		{
			power_sum[i] += static_cast<double>(h_power[i]);
		}
	}

	std::vector<float> BuildPowerDbImage(const std::vector<double>& power_sum)
	{
		std::vector<float> power_db(power_sum.size(), -200.0f);
		double peak = 0.0;
		for (double value : power_sum)
		{
			peak = std::max(peak, std::sqrt(std::max(value, 0.0)));
		}

		if (peak <= 0.0)
		{
			return power_db;
		}

		for (size_t i = 0; i < power_sum.size(); ++i)
		{
			const double magnitude = std::sqrt(std::max(power_sum[i], 0.0)) / peak;
			power_db[i] = 20.0f * std::log10(std::max(static_cast<float>(magnitude), 1.0e-20f));
		}
		return power_db;
	}

	void TryRenderDopplerColorPngWithPython(
		const std::filesystem::path& power_db_path,
		int width,
		int height,
		const std::filesystem::path& output_png,
		float display_min_db,
		float display_max_db)
	{
#ifdef _WIN32
		const std::filesystem::path script_path = std::filesystem::absolute("render_power_doppler_png.py");
		if (!std::filesystem::exists(script_path))
		{
			throw std::runtime_error("Missing Doppler PNG renderer script: " + script_path.string());
		}
		if (!std::filesystem::exists(power_db_path))
		{
			throw std::runtime_error("Missing power_db input for Doppler PNG render: " + power_db_path.string());
		}

		std::ostringstream command;
		command
			<< "py -3 "
			<< "\"" << script_path.string() << "\" "
			<< "--power-db "
			<< "\"" << power_db_path.string() << "\" "
			<< "--width " << width << " "
			<< "--height " << height << " "
			<< "--output-png "
			<< "\"" << output_png.string() << "\" "
			<< "--display-min-db " << display_min_db << " "
			<< "--display-max-db " << display_max_db << " "
			<< "--x-min-mm " << pc.x_start * 1.0e3f << " "
			<< "--x-max-mm " << pc.x_end * 1.0e3f << " "
			<< "--z-min-mm " << pc.z_start * 1.0e3f << " "
			<< "--z-max-mm " << pc.z_end * 1.0e3f;

		const int exit_code = std::system(command.str().c_str());
		if (exit_code != 0)
		{
			throw std::runtime_error(
				"Doppler color PNG render failed with exit code " +
				std::to_string(exit_code) + ": " + output_png.string());
		}
#else
		(void)power_db_path;
		(void)width;
		(void)height;
		(void)output_png;
		(void)display_min_db;
		(void)display_max_db;
#endif
	}

	int RunZenodo7883227RsvdPowerDopplerBinSerial(
		const std::filesystem::path& bin_dir,
		const std::filesystem::path& output_prefix,
		int cutoff_start,
		int cutoff_end,
		int max_files)
	{
		float2* h_iq = nullptr;
		float2* d_iq = nullptr;
		float2* d_iq_cache = nullptr;
		float2* d_iq_filtered = nullptr;
		float* d_power = nullptr;

		try
		{
			ConfigureZenodo7883227PowerDoppler();

			const int total_frames = 800;
			const int pixels = pc.img_nx * pc.img_nz;
			const size_t iq_elements = static_cast<size_t>(pc.seq) * pc.channels * pc.datalen * total_frames;
			const size_t iq_bytes = iq_elements * sizeof(float2);
			const size_t iq_cache_elements = static_cast<size_t>(pixels) * total_frames;
			const size_t iq_cache_bytes = iq_cache_elements * sizeof(float2);
			const size_t power_bytes = static_cast<size_t>(pixels) * sizeof(float);

			const std::vector<std::filesystem::path> bin_files = ListBinFiles(bin_dir, max_files);
			std::vector<double> power_sum(static_cast<size_t>(pixels), 0.0);
			std::vector<float> h_power(static_cast<size_t>(pixels), 0.0f);
			BEAMFORMER beamform;

			RsvdFilterConfig rsvd_config{};
			rsvd_config.rank = std::max(cutoff_start - 1, 1);
			rsvd_config.oversampling = 32;
			rsvd_config.power_iterations = 2;
			rsvd_config.cutoff_start = cutoff_start;
			rsvd_config.cutoff_end = cutoff_end;

			CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&h_iq), iq_bytes, cudaHostAllocDefault));
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_iq), iq_bytes));
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_iq_cache), iq_cache_bytes));
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_iq_filtered), iq_cache_bytes));
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_power), power_bytes));

			const auto wall_start = std::chrono::high_resolution_clock::now();
			double read_total_s = 0.0;
			double gpu_total_s = 0.0;
			double svd_filter_total_s = 0.0;

			for (size_t file_idx = 0; file_idx < bin_files.size(); ++file_idx)
			{
				std::cout << "[Info] (" << (file_idx + 1) << "/" << bin_files.size()
					<< ") Process " << bin_files[file_idx].filename().string() << std::endl;

				const auto read_start = std::chrono::high_resolution_clock::now();
				ReadComplexDataToPinnedBuffer(bin_files[file_idx], h_iq, iq_elements);
				const auto read_stop = std::chrono::high_resolution_clock::now();
				read_total_s += std::chrono::duration<double>(read_stop - read_start).count();

				const auto gpu_start = std::chrono::high_resolution_clock::now();
				CUDA_CHECK(cudaMemcpy(d_iq, h_iq, iq_bytes, cudaMemcpyHostToDevice));
				beamform.BeamformIQCacheSharedRectChannelOuterCore(d_iq, total_frames, d_iq_cache);

				RsvdFilterStats svd_stats{};
				RsvdFilterIqCache(
					d_iq_cache,
					pixels,
					total_frames,
					rsvd_config,
					d_iq_filtered,
					&svd_stats);
				svd_filter_total_s += static_cast<double>(svd_stats.elapsed_ms) * 1.0e-3;

				PowerDopplerFromIqCache(d_iq_filtered, pixels, total_frames, d_power);
				CUDA_CHECK(cudaDeviceSynchronize());
				CUDA_CHECK(cudaMemcpy(
					h_power.data(),
					d_power,
					power_bytes,
					cudaMemcpyDeviceToHost));
				const auto gpu_stop = std::chrono::high_resolution_clock::now();
				gpu_total_s += std::chrono::duration<double>(gpu_stop - gpu_start).count();

				AccumulatePowerBlock(power_sum, h_power.data(), static_cast<size_t>(pixels));
			}

			const auto wall_stop = std::chrono::high_resolution_clock::now();
			const double wall_total_s = std::chrono::duration<double>(wall_stop - wall_start).count();

			const std::filesystem::path power_sum_path = output_prefix.string() + "_power_sum.bin";
			const std::filesystem::path power_db_path = output_prefix.string() + "_power_db.bin";
			const std::filesystem::path png_path = output_prefix.string() + "_db40to0_red.png";
			const std::filesystem::path summary_path = output_prefix.string() + ".txt";

			std::vector<float> power_sum_float(power_sum.size());
			for (size_t i = 0; i < power_sum.size(); ++i)
			{
				power_sum_float[i] = static_cast<float>(power_sum[i]);
			}
			const std::vector<float> power_db = BuildPowerDbImage(power_sum);

			SaveFloatBinary(power_sum_path, power_sum_float);
			SaveFloatBinary(power_db_path, power_db);
			TryRenderDopplerColorPngWithPython(
				power_db_path,
				pc.img_nx,
				pc.img_nz,
				png_path,
				-40.0f,
				0.0f);

			std::filesystem::create_directories(summary_path.parent_path());
			std::ofstream summary(summary_path, std::ios::out | std::ios::trunc);
			if (!summary.is_open())
			{
				throw std::runtime_error("Cannot create summary file: " + summary_path.string());
			}
			summary
				<< "mode=serial_bin\n"
				<< "bin_dir=" << bin_dir.string() << "\n"
				<< "processed_files=" << bin_files.size() << "\n"
				<< "svd_method=covariance_exact\n"
				<< "cutoff_start=" << cutoff_start << "\n"
				<< "cutoff_end=" << cutoff_end << "\n"
				<< "power_sum_bin=" << power_sum_path.string() << "\n"
				<< "power_db_bin=" << power_db_path.string() << "\n"
				<< "png=" << png_path.string() << "\n"
				<< "read_total_s=" << read_total_s << "\n"
				<< "gpu_total_s=" << gpu_total_s << "\n"
				<< "svd_filter_total_s=" << svd_filter_total_s << "\n"
				<< "wall_total_s=" << wall_total_s << "\n";
			for (const auto& file : bin_files)
			{
				summary << file.filename().string() << "\n";
			}

			std::cout << std::fixed << std::setprecision(3)
				<< "[Info] Serial bin pipeline done: files=" << bin_files.size()
				<< ", wall=" << wall_total_s << " s"
				<< ", read=" << read_total_s << " s"
				<< ", gpu_total=" << gpu_total_s << " s"
				<< ", svd_filter=" << svd_filter_total_s << " s"
				<< std::endl;
			std::cout << "[Info] Saved power sum to " << power_sum_path.string() << std::endl;
			std::cout << "[Info] Saved power dB to " << power_db_path.string() << std::endl;
			std::cout << "[Info] Saved Doppler PNG to " << png_path.string() << std::endl;
			std::cout << "[Info] Saved summary to " << summary_path.string() << std::endl;

			CUDA_CHECK(cudaFree(d_power));
			d_power = nullptr;
			CUDA_CHECK(cudaFree(d_iq_filtered));
			d_iq_filtered = nullptr;
			CUDA_CHECK(cudaFree(d_iq_cache));
			d_iq_cache = nullptr;
			CUDA_CHECK(cudaFree(d_iq));
			d_iq = nullptr;
			CUDA_CHECK(cudaFreeHost(h_iq));
			h_iq = nullptr;
			return 0;
		}
		catch (const std::exception& ex)
		{
			if (d_power != nullptr) { cudaFree(d_power); }
			if (d_iq_filtered != nullptr) { cudaFree(d_iq_filtered); }
			if (d_iq_cache != nullptr) { cudaFree(d_iq_cache); }
			if (d_iq != nullptr) { cudaFree(d_iq); }
			if (h_iq != nullptr) { cudaFreeHost(h_iq); }
			std::cerr << "[Error] " << ex.what() << std::endl;
			return 1;
		}
	}

	void PrintUsage()
	{
		std::cerr << "Usage: LinearArrayCPWC_OPT_manual.exe "
			<< "doppler7883227_svd_powdop_binserial <bin_dir> <output_prefix> "
			<< "[cutoff_start] [cutoff_end] [max_files]"
			<< std::endl;
	}

	int RunCliMode(int argc, char** argv)
	{
		if (argc < 2)
		{
			PrintUsage();
			return 1;
		}

		const std::string mode = argv[1];
		if (mode != "doppler7883227_svd_powdop_binserial" &&
			mode != "doppler7883227_svd_powdop_binbatch" &&
			mode != "doppler7883227_rsvd_powdop_binserial" &&
			mode != "doppler7883227_rsvd_powdop_binbatch")
		{
			std::cerr << "[Error] Unsupported mode in quality project: " << mode << std::endl;
			PrintUsage();
			return 1;
		}
		if (argc < 4)
		{
			PrintUsage();
			return 1;
		}

		const std::filesystem::path bin_dir = argv[2];
		const std::filesystem::path output_prefix = argv[3];
		const int cutoff_start = (argc >= 5) ? std::stoi(argv[4]) : 50;
		const int cutoff_end = (argc >= 6) ? std::stoi(argv[5]) : 800;
		const int max_files = (argc >= 7) ? std::stoi(argv[6]) : 75;
		return RunZenodo7883227RsvdPowerDopplerBinSerial(
			bin_dir,
			output_prefix,
			cutoff_start,
			cutoff_end,
			max_files);
	}
}

int main(int argc, char** argv)
{
	return RunCliMode(argc, argv);
}
