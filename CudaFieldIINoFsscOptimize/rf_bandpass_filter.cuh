#ifndef RF_BANDPASS_FILTER_CUH
#define RF_BANDPASS_FILTER_CUH

#include <cuda_runtime.h>

void ApplyRfBandpassFilter(
	const float* d_input,
	float* d_output,
	const float* d_coefficients,
	int traces,
	int samples,
	int taps,
	cudaStream_t stream = 0);

#endif
