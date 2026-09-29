#pragma once

#include "cuda_runtime.h"

#include <cstddef>

class FieldIIRfToIqPreprocessor
{
public:
	FieldIIRfToIqPreprocessor(int samples, int channels, int seq, int frames);
	~FieldIIRfToIqPreprocessor();

	FieldIIRfToIqPreprocessor(const FieldIIRfToIqPreprocessor&) = delete;
	FieldIIRfToIqPreprocessor& operator=(const FieldIIRfToIqPreprocessor&) = delete;

	void RfToIq(const float* d_rf, float2* d_iq, cudaStream_t stream = 0);
	void RfToHilbertFrameMajor(const float* d_rf, cudaStream_t stream = 0);
	const float* HilbertFrameMajor() const;

	std::size_t WorkspaceBytes() const;

private:
	int samples_ = 0;
	int channels_ = 0;
	int seq_ = 0;
	int frames_ = 0;
	int padded_samples_ = 0;
	void* d_half_spectrum_ = nullptr;
	void* d_padded_time_ = nullptr;
	void* r2c_plan_ = nullptr;
	void* inverse_plan_ = nullptr;
};

void StoreFieldIIRfHilbertBeamformerLayout(
	const float* d_rf,
	const float* d_hilbert_frame_major,
	float2* d_iq,
	int samples,
	int padded_samples,
	int channels,
	int seq,
	int frames,
	float hilbert_scale,
	cudaStream_t stream = 0);
