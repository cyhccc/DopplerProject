#include "motion_correction.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace
{
	float Magnitude(const float2& value)
	{
		return std::sqrt(value.x * value.x + value.y * value.y);
	}

	float2 Add(const float2& a, const float2& b)
	{
		float2 out{};
		out.x = a.x + b.x;
		out.y = a.y + b.y;
		return out;
	}

	float2 Mul(const float2& a, float scale)
	{
		float2 out{};
		out.x = a.x * scale;
		out.y = a.y * scale;
		return out;
	}

	float2 SampleBilinearZero(const std::vector<float2>& input, int width, int height, float x, float z)
	{
		if (x < 0.0f || z < 0.0f || x > static_cast<float>(width - 1) || z > static_cast<float>(height - 1))
		{
			return float2{};
		}
		const int x0 = static_cast<int>(std::floor(x));
		const int z0 = static_cast<int>(std::floor(z));
		const int x1 = x0 + 1;
		const int z1 = z0 + 1;
		const float fx = x - static_cast<float>(x0);
		const float fz = z - static_cast<float>(z0);

		const auto at = [&](int xi, int zi) -> float2 {
			if (xi < 0 || zi < 0 || xi >= width || zi >= height)
			{
				return float2{};
			}
			return input[static_cast<size_t>(zi) * width + xi];
		};

		const float2 v00 = at(x0, z0);
		const float2 v10 = at(x1, z0);
		const float2 v01 = at(x0, z1);
		const float2 v11 = at(x1, z1);
		const float2 top = Add(Mul(v00, 1.0f - fx), Mul(v10, fx));
		const float2 bottom = Add(Mul(v01, 1.0f - fx), Mul(v11, fx));
		return Add(Mul(top, 1.0f - fz), Mul(bottom, fz));
	}

	void ValidateConfig(int width, int height, const MotionCorrectionConfig& config)
	{
		if (width <= 0 || height <= 0)
		{
			throw std::invalid_argument("Motion correction image dimensions must be positive.");
		}
		if (config.search_radius_x < 0 || config.search_radius_z < 0)
		{
			throw std::invalid_argument("Motion correction search radius must be non-negative.");
		}
		if (config.stride_x <= 0 || config.stride_z <= 0)
		{
			throw std::invalid_argument("Motion correction strides must be positive.");
		}
		if (config.roi_x0 < 0 || config.roi_z0 < 0 ||
			config.roi_x1 >= width || config.roi_z1 >= height ||
			config.roi_x1 < config.roi_x0 || config.roi_z1 < config.roi_z0)
		{
			throw std::invalid_argument("Motion correction ROI is outside the image.");
		}
	}

	float NormalizedCorrelationForShift(
		const std::vector<float2>& reference,
		const std::vector<float2>& moving,
		int width,
		int height,
		const MotionCorrectionConfig& config,
		int dx,
		int dz)
	{
		double sum_ref = 0.0;
		double sum_mov = 0.0;
		double sum_ref2 = 0.0;
		double sum_mov2 = 0.0;
		double sum_cross = 0.0;
		int count = 0;

		for (int z = config.roi_z0; z <= config.roi_z1; z += config.stride_z)
		{
			const int mz = z + dz;
			if (mz < 0 || mz >= height)
			{
				continue;
			}
			for (int x = config.roi_x0; x <= config.roi_x1; x += config.stride_x)
			{
				const int mx = x + dx;
				if (mx < 0 || mx >= width)
				{
					continue;
				}
				const float ref = Magnitude(reference[static_cast<size_t>(z) * width + x]);
				const float mov = Magnitude(moving[static_cast<size_t>(mz) * width + mx]);
				sum_ref += ref;
				sum_mov += mov;
				sum_ref2 += static_cast<double>(ref) * ref;
				sum_mov2 += static_cast<double>(mov) * mov;
				sum_cross += static_cast<double>(ref) * mov;
				++count;
			}
		}

		if (count < 16)
		{
			return -1.0f;
		}
		const double inv_count = 1.0 / static_cast<double>(count);
		const double cov = sum_cross - sum_ref * sum_mov * inv_count;
		const double var_ref = sum_ref2 - sum_ref * sum_ref * inv_count;
		const double var_mov = sum_mov2 - sum_mov * sum_mov * inv_count;
		const double denom = std::sqrt(std::max(var_ref * var_mov, 0.0));
		if (denom <= 1.0e-20)
		{
			return -1.0f;
		}
		return static_cast<float>(cov / denom);
	}

	std::vector<MotionShift> SmoothShifts(const std::vector<MotionShift>& raw, int radius)
	{
		if (radius <= 0 || raw.empty())
		{
			return raw;
		}
		std::vector<MotionShift> smoothed(raw.size());
		for (size_t i = 0; i < raw.size(); ++i)
		{
			const size_t begin = (i > static_cast<size_t>(radius)) ? i - static_cast<size_t>(radius) : 0;
			const size_t end = std::min(raw.size() - 1, i + static_cast<size_t>(radius));
			double dx = 0.0;
			double dz = 0.0;
			double score = 0.0;
			int count = 0;
			for (size_t j = begin; j <= end; ++j)
			{
				dx += raw[j].dx;
				dz += raw[j].dz;
				score += raw[j].score;
				++count;
			}
			smoothed[i].dx = static_cast<float>(dx / count);
			smoothed[i].dz = static_cast<float>(dz / count);
			smoothed[i].score = static_cast<float>(score / count);
		}
		return smoothed;
	}

	void UpdateShiftStats(MotionCorrectionSummary& summary)
	{
		const std::vector<MotionShift>& shifts = summary.smoothed_shifts;
		if (shifts.empty())
		{
			return;
		}
		summary.min_dx = summary.max_dx = shifts.front().dx;
		summary.min_dz = summary.max_dz = shifts.front().dz;
		double abs_dx = 0.0;
		double abs_dz = 0.0;
		for (const MotionShift& shift : shifts)
		{
			summary.min_dx = std::min(summary.min_dx, shift.dx);
			summary.max_dx = std::max(summary.max_dx, shift.dx);
			summary.min_dz = std::min(summary.min_dz, shift.dz);
			summary.max_dz = std::max(summary.max_dz, shift.dz);
			abs_dx += std::abs(shift.dx);
			abs_dz += std::abs(shift.dz);
		}
		summary.mean_abs_dx = static_cast<float>(abs_dx / shifts.size());
		summary.mean_abs_dz = static_cast<float>(abs_dz / shifts.size());
	}
}

MotionShift EstimateGlobalShiftFromEnvelope(
	const std::vector<float2>& reference,
	const std::vector<float2>& moving,
	int width,
	int height,
	const MotionCorrectionConfig& config)
{
	if (reference.size() != static_cast<size_t>(width) * height ||
		moving.size() != static_cast<size_t>(width) * height)
	{
		throw std::invalid_argument("Motion correction frame size mismatch.");
	}
	ValidateConfig(width, height, config);

	MotionShift best{};
	best.score = -std::numeric_limits<float>::infinity();
	for (int dz = -config.search_radius_z; dz <= config.search_radius_z; ++dz)
	{
		for (int dx = -config.search_radius_x; dx <= config.search_radius_x; ++dx)
		{
			const float score = NormalizedCorrelationForShift(reference, moving, width, height, config, dx, dz);
			if (score > best.score)
			{
				best.dx = static_cast<float>(dx);
				best.dz = static_cast<float>(dz);
				best.score = score;
			}
		}
	}
	return best;
}

void WarpIqFrameBilinear(
	const std::vector<float2>& input,
	std::vector<float2>& output,
	int width,
	int height,
	float dx,
	float dz)
{
	if (input.size() != static_cast<size_t>(width) * height)
	{
		throw std::invalid_argument("Motion correction warp input size mismatch.");
	}
	output.resize(input.size());
	for (int z = 0; z < height; ++z)
	{
		for (int x = 0; x < width; ++x)
		{
			output[static_cast<size_t>(z) * width + x] =
				SampleBilinearZero(input, width, height, static_cast<float>(x) + dx, static_cast<float>(z) + dz);
		}
	}
}

MotionCorrectionSummary ApplyGlobalMotionCorrection(
	std::vector<float2>& iq_cache,
	int width,
	int height,
	int frames,
	const MotionCorrectionConfig& config)
{
	if (frames <= 0)
	{
		throw std::invalid_argument("Motion correction frame count must be positive.");
	}
	const size_t pixels = static_cast<size_t>(width) * height;
	if (iq_cache.size() != pixels * static_cast<size_t>(frames))
	{
		throw std::invalid_argument("Motion correction IQ cache size mismatch.");
	}
	ValidateConfig(width, height, config);

	const auto start = std::chrono::high_resolution_clock::now();
	MotionCorrectionSummary summary{};

	std::vector<float2> reference(pixels, float2{});
	for (int frame = 0; frame < frames; ++frame)
	{
		const float2* src = iq_cache.data() + pixels * static_cast<size_t>(frame);
		for (size_t i = 0; i < pixels; ++i)
		{
			reference[i].x += Magnitude(src[i]);
		}
	}
	const float inv_frames = 1.0f / static_cast<float>(frames);
	for (float2& value : reference)
	{
		value.x *= inv_frames;
		value.y = 0.0f;
	}

	summary.raw_shifts.resize(static_cast<size_t>(frames));
	std::vector<float2> moving(pixels);
	for (int frame = 0; frame < frames; ++frame)
	{
		const float2* src = iq_cache.data() + pixels * static_cast<size_t>(frame);
		std::copy(src, src + pixels, moving.begin());
		summary.raw_shifts[static_cast<size_t>(frame)] =
			EstimateGlobalShiftFromEnvelope(reference, moving, width, height, config);
	}
	summary.smoothed_shifts = SmoothShifts(summary.raw_shifts, config.smooth_radius);

	std::vector<float2> input_frame(pixels);
	std::vector<float2> output_frame(pixels);
	for (int frame = 0; frame < frames; ++frame)
	{
		float2* dst = iq_cache.data() + pixels * static_cast<size_t>(frame);
		std::copy(dst, dst + pixels, input_frame.begin());
		const MotionShift& shift = summary.smoothed_shifts[static_cast<size_t>(frame)];
		WarpIqFrameBilinear(input_frame, output_frame, width, height, shift.dx, shift.dz);
		std::copy(output_frame.begin(), output_frame.end(), dst);
	}

	UpdateShiftStats(summary);
	summary.elapsed_s =
		std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
	return summary;
}
