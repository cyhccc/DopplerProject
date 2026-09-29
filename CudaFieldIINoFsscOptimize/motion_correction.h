#ifndef MOTION_CORRECTION_H
#define MOTION_CORRECTION_H

#include <cuda_runtime.h>

#include <vector>

struct MotionCorrectionConfig
{
	bool enabled = false;
	int search_radius_x = 8;
	int search_radius_z = 6;
	int roi_x0 = 0;
	int roi_x1 = 0;
	int roi_z0 = 0;
	int roi_z1 = 0;
	int stride_x = 4;
	int stride_z = 4;
	int smooth_radius = 2;
};

struct MotionShift
{
	float dx = 0.0f;
	float dz = 0.0f;
	float score = 0.0f;
};

struct MotionCorrectionSummary
{
	std::vector<MotionShift> raw_shifts;
	std::vector<MotionShift> smoothed_shifts;
	float min_dx = 0.0f;
	float max_dx = 0.0f;
	float min_dz = 0.0f;
	float max_dz = 0.0f;
	float mean_abs_dx = 0.0f;
	float mean_abs_dz = 0.0f;
	double elapsed_s = 0.0;
};

MotionShift EstimateGlobalShiftFromEnvelope(
	const std::vector<float2>& reference,
	const std::vector<float2>& moving,
	int width,
	int height,
	const MotionCorrectionConfig& config);

void WarpIqFrameBilinear(
	const std::vector<float2>& input,
	std::vector<float2>& output,
	int width,
	int height,
	float dx,
	float dz);

MotionCorrectionSummary ApplyGlobalMotionCorrection(
	std::vector<float2>& iq_cache,
	int width,
	int height,
	int frames,
	const MotionCorrectionConfig& config);

#endif
