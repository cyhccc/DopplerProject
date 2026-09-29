#pragma once

#include "cuda_runtime.h"

#include <cstddef>
#include <vector>

struct RsvdFilterConfig
{
	int rank = 4;
	int oversampling = 32;
	int power_iterations = 2;
	int cutoff_start = 5;
	int cutoff_end = 0; // 0 means n_frames.
};

struct RsvdFilterStats
{
	float elapsed_ms = 0.0f;
	std::size_t device_workspace_bytes = 0;
	std::size_t host_workspace_bytes = 0;
	std::vector<float> singular_values;
};

struct SsmCutoffConfig
{
	int max_components = 800;
	int min_tissue = 5;
	int min_flow = 20;
};

struct SsmCutoffResult
{
	int max_components = 0;
	int tissue_component_count = 0;
	int flow_component_end = 0;
	int cutoff_start = 1;
	int cutoff_end = 0;
	float fit_score = 0.0f;
	float elapsed_ms = 0.0f;
	std::size_t device_workspace_bytes = 0;
	std::size_t host_workspace_bytes = 0;
	std::vector<float> singular_values;
	std::vector<float> ssm_matrix;
};

// Filters the beamformed IQ cache on the GPU.
//
// Input/output layout is cuSOLVER/cuBLAS-friendly column-major Casorati:
// logical X(pixel, frame) is stored at cache[pixel + frame * pixels].
// This matches BEAMFORMER::BeamformIQCacheSharedRectChannelOuterCore.
//
// The implementation uses an exact temporal SVD path based on the Hermitian
// Gram matrix X^H X. The historical "Rsvd" names are kept for CLI/source
// compatibility with older scripts.
void RsvdFilterIqCache(
	const float2* d_iq_cache,
	int pixels,
	int n_frames,
	const RsvdFilterConfig& config,
	float2* d_filtered_iq_cache,
	RsvdFilterStats* stats = nullptr,
	cudaStream_t stream = 0);

// Estimates an SSM-based SVD cutoff from the beamformed IQ cache.
//
// The SSM itself is formed on the GPU:
//   1. exact covariance eigendecomposition of X^H X,
//   2. spatial singular vectors U = X * V / sigma,
//   3. SSM = abs(corr(abs(U))).
//
// A lightweight two-square boundary fit is then applied to the resulting
// max_components x max_components SSM to estimate
//   tissue = [1, cutoff_start-1],
//   flow   = [cutoff_start, cutoff_end],
//   noise  = [cutoff_end+1, n_frames].
void EstimateSsmCutoffFromIqCache(
	const float2* d_iq_cache,
	int pixels,
	int n_frames,
	const SsmCutoffConfig& config,
	SsmCutoffResult* result,
	cudaStream_t stream = 0);

// True randomized SVD path: estimates the leading temporal singular vectors
// and subtracts only the top-k low-rank clutter component:
// X_filtered = X - X * V_k * V_k^H.
void RsvdTopKSubtractIqCache(
	const float2* d_iq_cache,
	int pixels,
	int n_frames,
	int remove_rank,
	int oversampling,
	int power_iterations,
	float2* d_filtered_iq_cache,
	RsvdFilterStats* stats = nullptr,
	cudaStream_t stream = 0);

// Computes power Doppler from a column-major Casorati IQ cache:
// power(pixel) = sum_frame |IQ(pixel, frame)|^2
void PowerDopplerFromIqCache(
	const float2* d_iq_cache,
	int pixels,
	int n_frames,
	float* d_power,
	cudaStream_t stream = 0);
