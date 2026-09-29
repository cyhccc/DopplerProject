#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "PARAMs.h"

class BEAMFORMER
{
public:
	BEAMFORMER();
	~BEAMFORMER();

	// Raw beamformed IQ cache for post-beamforming SVD.
	// d_iq_cache is a column-major Casorati matrix:
	// logical X(pixel, frame) is stored at d_iq_cache[pixel + frame * pixels].
	void BeamformIQCacheSharedRectChannelOuterCore(
		const float2* d_sig,
		int n_frames,
		float2* d_iq_cache);
};

__global__ void Kernel_Linear_Array_CPWC_IQCache_Row32x8_SharedRectChannelOuter(
	const float2* __restrict__ d_sig,
	float2* d_iq_cache,
	int sum_pixel,
	int channels,
	int datalen,
	int seq,
	int n_frames,
	PARAMs pc
);
