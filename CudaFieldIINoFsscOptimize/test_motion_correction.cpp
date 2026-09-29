#include "motion_correction.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace
{
	void Require(bool condition, const char* message)
	{
		if (!condition)
		{
			throw std::runtime_error(message);
		}
	}

	float2 Make(float x, float y)
	{
		float2 v{};
		v.x = x;
		v.y = y;
		return v;
	}

	void AddBlob(std::vector<float2>& frame, int width, int height, int cx, int cz)
	{
		for (int z = 0; z < height; ++z)
		{
			for (int x = 0; x < width; ++x)
			{
				const float dx = static_cast<float>(x - cx);
				const float dz = static_cast<float>(z - cz);
				const float value = std::exp(-(dx * dx + dz * dz) / 18.0f);
				frame[static_cast<size_t>(z) * width + x] = Make(value, 0.5f * value);
			}
		}
	}
}

int main()
{
	const int width = 32;
	const int height = 24;
	const int pixels = width * height;
	std::vector<float2> reference(static_cast<size_t>(pixels));
	std::vector<float2> shifted(static_cast<size_t>(pixels));
	AddBlob(reference, width, height, 12, 10);
	AddBlob(shifted, width, height, 15, 8);

	MotionCorrectionConfig config{};
	config.enabled = true;
	config.search_radius_x = 6;
	config.search_radius_z = 6;
	config.roi_x0 = 0;
	config.roi_x1 = width - 1;
	config.roi_z0 = 0;
	config.roi_z1 = height - 1;
	config.stride_x = 1;
	config.stride_z = 1;
	config.smooth_radius = 0;

	const MotionShift shift = EstimateGlobalShiftFromEnvelope(reference, shifted, width, height, config);
	Require(std::abs(shift.dx - 3.0f) < 0.25f, "dx estimate should recover known +3 px shift");
	Require(std::abs(shift.dz + 2.0f) < 0.25f, "dz estimate should recover known -2 px shift");

	std::vector<float2> aligned(static_cast<size_t>(pixels));
	WarpIqFrameBilinear(shifted, aligned, width, height, shift.dx, shift.dz);
	const float2 ref_peak = reference[static_cast<size_t>(10) * width + 12];
	const float2 aligned_peak = aligned[static_cast<size_t>(10) * width + 12];
	Require(std::abs(ref_peak.x - aligned_peak.x) < 1.0e-3f, "aligned real peak should match reference");
	Require(std::abs(ref_peak.y - aligned_peak.y) < 1.0e-3f, "aligned imag peak should match reference");

	std::cout << "motion_correction_test_passed dx=" << shift.dx
		<< " dz=" << shift.dz
		<< " score=" << shift.score << "\n";
	return 0;
}
