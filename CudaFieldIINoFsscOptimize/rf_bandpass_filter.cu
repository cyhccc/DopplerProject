#include "rf_bandpass_filter.cuh"

#include "config.cuh"

#include <stdexcept>

namespace
{
	__global__ void RfBandpassCenteredFirKernel(
		const float* __restrict__ input,
		float* __restrict__ output,
		const float* __restrict__ coefficients,
		int traces,
		int samples,
		int taps)
	{
		const int sample = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
		const int trace = static_cast<int>(blockIdx.y);
		if (sample >= samples || trace >= traces)
		{
			return;
		}

		const int center = taps / 2;
		const size_t base = static_cast<size_t>(trace) * samples;
		float sum = 0.0f;
		for (int tap = 0; tap < taps; ++tap)
		{
			const int source_sample = sample + tap - center;
			if (source_sample >= 0 && source_sample < samples)
			{
				sum = fmaf(coefficients[tap], input[base + static_cast<size_t>(source_sample)], sum);
			}
		}
		output[base + static_cast<size_t>(sample)] = sum;
	}
}

void ApplyRfBandpassFilter(
	const float* d_input,
	float* d_output,
	const float* d_coefficients,
	int traces,
	int samples,
	int taps,
	cudaStream_t stream)
{
	if (d_input == nullptr || d_output == nullptr || d_coefficients == nullptr)
	{
		throw std::invalid_argument("Bandpass filter input/output/coefficient pointer must not be null.");
	}
	if (traces <= 0 || samples <= 0 || taps <= 0 || (taps % 2) == 0)
	{
		throw std::invalid_argument("Bandpass filter expects positive traces/samples and odd tap count.");
	}

	const dim3 threads(128, 1, 1);
	const dim3 blocks((samples + threads.x - 1) / threads.x, traces, 1);
	RfBandpassCenteredFirKernel<<<blocks, threads, 0, stream>>>(
		d_input,
		d_output,
		d_coefficients,
		traces,
		samples,
		taps);
	CUDA_CHECK(cudaGetLastError());
}
