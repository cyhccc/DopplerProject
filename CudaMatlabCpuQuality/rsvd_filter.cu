#include "rsvd_filter.cuh"

#include "cuda_intellisense.cuh"

#include "cublas_v2.h"
#include "cuComplex.h"
#include "cusolverDn.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
using ComplexType = cuComplex;
using SingularReal = float;

static_assert(sizeof(ComplexType) == sizeof(float2), "cuComplex and float2 must have the same layout.");

[[noreturn]] void ThrowError(const char* api, const char* call, const char* file, int line, int code)
{
	std::ostringstream oss;
	oss << api << " call failed at " << file << ":" << line << "\n"
		<< "  call: " << call << "\n"
		<< "  code: " << code;
	throw std::runtime_error(oss.str());
}

void CheckCuda(cudaError_t status, const char* call, const char* file, int line)
{
	if (status != cudaSuccess)
	{
		std::ostringstream oss;
		oss << "CUDA call failed at " << file << ":" << line << "\n"
			<< "  call: " << call << "\n"
			<< "  error: " << cudaGetErrorString(status);
		throw std::runtime_error(oss.str());
	}
}

void CheckCublas(cublasStatus_t status, const char* call, const char* file, int line)
{
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		ThrowError("cuBLAS", call, file, line, static_cast<int>(status));
	}
}

void CheckCusolver(cusolverStatus_t status, const char* call, const char* file, int line)
{
	if (status != CUSOLVER_STATUS_SUCCESS)
	{
		ThrowError("cuSOLVER", call, file, line, static_cast<int>(status));
	}
}

#define SVD_CUDA_CHECK(call) CheckCuda((call), #call, __FILE__, __LINE__)
#define SVD_CUBLAS_CHECK(call) CheckCublas((call), #call, __FILE__, __LINE__)
#define SVD_CUSOLVER_CHECK(call) CheckCusolver((call), #call, __FILE__, __LINE__)

int ResolveCutoffEnd(const RsvdFilterConfig& config, int n_frames)
{
	return config.cutoff_end > 0 ? config.cutoff_end : n_frames;
}

__global__ void Kernel_PowerDopplerFromIqCache(
	const float2* __restrict__ d_iq_cache,
	int pixels,
	int n_frames,
	float* __restrict__ d_power)
{
	const int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pixel_idx >= pixels)
	{
		return;
	}

	float power = 0.0f;
	for (int frame_idx = 0; frame_idx < n_frames; ++frame_idx)
	{
		const float2 iq = d_iq_cache[static_cast<size_t>(frame_idx) * pixels + pixel_idx];
		power += iq.x * iq.x + iq.y * iq.y;
	}

	d_power[pixel_idx] = power;
}

void ValidateConfig(const RsvdFilterConfig& config, int pixels, int n_frames)
{
	if (pixels <= 0 || n_frames <= 0)
	{
		throw std::invalid_argument("pixels and n_frames must be positive.");
	}
	if (pixels > std::numeric_limits<int>::max() / n_frames)
	{
		throw std::invalid_argument("pixels * n_frames overflows int indexing.");
	}

	const int cutoff_end = ResolveCutoffEnd(config, n_frames);
	if (config.cutoff_start < 1 || cutoff_end < config.cutoff_start || cutoff_end > n_frames)
	{
		std::ostringstream oss;
		oss << "Invalid SVD cutoff [" << config.cutoff_start << ", " << cutoff_end
			<< "] for n_frames=" << n_frames << ".";
		throw std::invalid_argument(oss.str());
	}
}

void ValidateTopKConfig(int pixels, int n_frames, int remove_rank, int oversampling, int power_iterations)
{
	if (pixels <= 0 || n_frames <= 0)
	{
		throw std::invalid_argument("pixels and n_frames must be positive.");
	}
	if (pixels > std::numeric_limits<int>::max() / n_frames)
	{
		throw std::invalid_argument("pixels * n_frames overflows int indexing.");
	}
	if (remove_rank <= 0 || oversampling < 0 || power_iterations < 0)
	{
		throw std::invalid_argument("remove_rank must be positive; oversampling and power_iterations must be non-negative.");
	}

	const int min_mn = std::min(pixels, n_frames);
	if (remove_rank + oversampling >= min_mn)
	{
		std::ostringstream oss;
		oss << "True RSVD requires remove_rank + oversampling < min(pixels, n_frames), got "
			<< remove_rank << " + " << oversampling << " >= " << min_mn << ".";
		throw std::invalid_argument(oss.str());
	}
}

void ValidateSsmConfig(const SsmCutoffConfig& config, int pixels, int n_frames)
{
	if (pixels <= 0 || n_frames <= 0)
	{
		throw std::invalid_argument("pixels and n_frames must be positive.");
	}
	if (pixels > std::numeric_limits<int>::max() / n_frames)
	{
		throw std::invalid_argument("pixels * n_frames overflows int indexing.");
	}
	if (config.max_components <= 0 || config.max_components > n_frames)
	{
		throw std::invalid_argument("SSM max_components must be in [1, n_frames].");
	}
	if (config.min_tissue <= 0 || config.min_flow <= 0)
	{
		throw std::invalid_argument("SSM min_tissue and min_flow must be positive.");
	}
	if (config.min_tissue + config.min_flow > config.max_components)
	{
		throw std::invalid_argument("SSM requires min_tissue + min_flow <= max_components.");
	}
}

__global__ void KernelFillFloat(float* data, int count, float value)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < count)
	{
		data[idx] = value;
	}
}

__global__ void KernelSpatialToEnvelopeDescending(
	const ComplexType* d_spatial_asc,
	const float* d_eigen_values,
	float* d_envelope_desc,
	int pixels,
	int n_frames,
	int cols)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int total = pixels * cols;
	if (idx >= total)
	{
		return;
	}

	const int out_col = idx / pixels;
	const int row = idx - out_col * pixels;
	const int asc_col = cols - 1 - out_col;
	const int eig_idx = n_frames - 1 - out_col;
	const float lambda = fmaxf(d_eigen_values[eig_idx], 0.0f);
	const float sigma = sqrtf(lambda);
	if (sigma > 1.0e-20f)
	{
		const ComplexType value = d_spatial_asc[static_cast<std::size_t>(asc_col) * pixels + row];
		d_envelope_desc[idx] = hypotf(value.x / sigma, value.y / sigma);
	}
	else
	{
		d_envelope_desc[idx] = 0.0f;
	}
}

__global__ void KernelCenterColumns(float* matrix, const float* column_sums, int pixels, int cols)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int total = pixels * cols;
	if (idx >= total)
	{
		return;
	}

	const int col = idx / pixels;
	const float mean = column_sums[col] / static_cast<float>(pixels);
	matrix[idx] -= mean;
}

__global__ void KernelNormalizeColumns(float* matrix, const float* norms, int pixels, int cols)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int total = pixels * cols;
	if (idx >= total)
	{
		return;
	}

	const int col = idx / pixels;
	const float denom = norms[col];
	if (denom > 1.0e-20f)
	{
		matrix[idx] /= denom;
	}
	else
	{
		matrix[idx] = 0.0f;
	}
}

__global__ void KernelAbsInPlace(float* data, int count)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < count)
	{
		data[idx] = fabsf(data[idx]);
	}
}

double RectSum(const std::vector<double>& integral, int stride, int r0, int r1, int c0, int c1)
{
	return integral[static_cast<std::size_t>(r1) * stride + c1]
		- integral[static_cast<std::size_t>(r0) * stride + c1]
		- integral[static_cast<std::size_t>(r1) * stride + c0]
		+ integral[static_cast<std::size_t>(r0) * stride + c0];
}

void FitTwoSquareModel(
	const std::vector<float>& ssm,
	int k,
	int min_tissue,
	int min_flow,
	int* tissue_count,
	int* flow_end,
	float* fit_score)
{
	if (tissue_count == nullptr || flow_end == nullptr || fit_score == nullptr)
	{
		throw std::invalid_argument("SSM fit output pointers must not be null.");
	}
	if (static_cast<int>(ssm.size()) != k * k)
	{
		throw std::invalid_argument("SSM matrix size mismatch in FitTwoSquareModel.");
	}

	double mean_s = 0.0;
	double sum_s2 = 0.0;
	for (float value : ssm)
	{
		mean_s += static_cast<double>(value);
	}
	const double n2 = static_cast<double>(k) * static_cast<double>(k);
	mean_s /= n2;
	for (float value : ssm)
	{
		const double centered = static_cast<double>(value) - mean_s;
		sum_s2 += centered * centered;
	}

	const int stride = k + 1;
	std::vector<double> integral(static_cast<std::size_t>(stride) * stride, 0.0);
	for (int r = 0; r < k; ++r)
	{
		double row_sum = 0.0;
		for (int c = 0; c < k; ++c)
		{
			row_sum += static_cast<double>(ssm[static_cast<std::size_t>(r) * k + c]);
			integral[static_cast<std::size_t>(r + 1) * stride + (c + 1)] =
				integral[static_cast<std::size_t>(r) * stride + (c + 1)] + row_sum;
		}
	}

	double best_score = -std::numeric_limits<double>::infinity();
	int best_tissue = min_tissue;
	int best_flow_end = std::min(k, min_tissue + min_flow);

	for (int tissue = min_tissue; tissue <= k - min_flow; ++tissue)
	{
		const double square1_sum = RectSum(integral, stride, 0, tissue, 0, tissue);
		for (int flow = tissue + min_flow; flow <= k; ++flow)
		{
			const int flow_count = flow - tissue;
			const double m = static_cast<double>(tissue) * tissue + static_cast<double>(flow_count) * flow_count;
			const double model_var = m - (m * m / n2);
			if (model_var <= 0.0)
			{
				continue;
			}

			const double square2_sum = RectSum(integral, stride, tissue, flow, tissue, flow);
			const double numerator = square1_sum + square2_sum - mean_s * m;
			const double denom = std::sqrt(std::max(sum_s2 * model_var, std::numeric_limits<double>::epsilon()));
			const double score = numerator / denom;
			if (score > best_score)
			{
				best_score = score;
				best_tissue = tissue;
				best_flow_end = flow;
			}
		}
	}

	*tissue_count = best_tissue;
	*flow_end = best_flow_end;
	*fit_score = static_cast<float>(best_score);
}

void ValidateSolverInfo(int info)
{
	if (info == 0)
	{
		return;
	}

	std::ostringstream oss;
	oss << "cusolverDnCheevd failed with info=" << info << ".";
	throw std::runtime_error(oss.str());
}

} // namespace

void RsvdFilterIqCache(
	const float2* d_iq_cache,
	int pixels,
	int n_frames,
	const RsvdFilterConfig& config,
	float2* d_filtered_iq_cache,
	RsvdFilterStats* stats,
	cudaStream_t stream)
{
	if (d_iq_cache == nullptr)
	{
		throw std::invalid_argument("d_iq_cache must not be null.");
	}
	if (d_filtered_iq_cache == nullptr)
	{
		throw std::invalid_argument("d_filtered_iq_cache must not be null.");
	}

	ValidateConfig(config, pixels, n_frames);

	if (stats != nullptr)
	{
		*stats = RsvdFilterStats{};
	}

	const int cutoff_end = ResolveCutoffEnd(config, n_frames);
	const int clutter_count = std::max(config.cutoff_start - 1, 0);
	const bool tail_filter = (cutoff_end == n_frames);
	const int keep_count = cutoff_end - config.cutoff_start + 1;
	const int projected_cols = tail_filter ? clutter_count : keep_count;
	const int total = pixels * n_frames;
	const std::size_t matrix_bytes = sizeof(ComplexType) * static_cast<std::size_t>(total);
	const std::size_t gram_elements = static_cast<std::size_t>(n_frames) * static_cast<std::size_t>(n_frames);
	const ComplexType* const d_a_base = reinterpret_cast<const ComplexType*>(d_iq_cache);
	ComplexType* const d_output = reinterpret_cast<ComplexType*>(d_filtered_iq_cache);

	if (tail_filter && clutter_count == 0)
	{
		SVD_CUDA_CHECK(cudaMemcpyAsync(
			d_filtered_iq_cache,
			d_iq_cache,
			sizeof(float2) * static_cast<std::size_t>(total),
			cudaMemcpyDeviceToDevice,
			stream));
		return;
	}

	cublasHandle_t cublas_handle = nullptr;
	cusolverDnHandle_t cusolver_handle = nullptr;
	ComplexType* d_gram = nullptr;
	float* d_eigen_values = nullptr;
	ComplexType* d_projected = nullptr;
	ComplexType* d_recon = nullptr;
	ComplexType* d_work = nullptr;
	int* d_info = nullptr;
	cudaEvent_t start = nullptr;
	cudaEvent_t stop = nullptr;
	int lwork = 0;

	const cuComplex alpha = make_cuComplex(1.0f, 0.0f);
	const cuComplex beta = make_cuComplex(0.0f, 0.0f);
	const cuComplex minus_one = make_cuComplex(-1.0f, 0.0f);

	auto cleanup = [&]() {
		if (stop != nullptr) { cudaEventDestroy(stop); }
		if (start != nullptr) { cudaEventDestroy(start); }
		if (d_info != nullptr) { cudaFree(d_info); }
		if (d_work != nullptr) { cudaFree(d_work); }
		if (d_recon != nullptr) { cudaFree(d_recon); }
		if (d_projected != nullptr) { cudaFree(d_projected); }
		if (d_eigen_values != nullptr) { cudaFree(d_eigen_values); }
		if (d_gram != nullptr) { cudaFree(d_gram); }
		if (cusolver_handle != nullptr) { cusolverDnDestroy(cusolver_handle); }
		if (cublas_handle != nullptr) { cublasDestroy(cublas_handle); }
	};

	try
	{
		SVD_CUBLAS_CHECK(cublasCreate(&cublas_handle));
		SVD_CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));
		SVD_CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));
		SVD_CUSOLVER_CHECK(cusolverDnSetStream(cusolver_handle, stream));

		SVD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gram), sizeof(ComplexType) * gram_elements));
		SVD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_eigen_values), sizeof(float) * static_cast<std::size_t>(n_frames)));
		SVD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_info), sizeof(int)));
		SVD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_recon), matrix_bytes));
		if (projected_cols > 0)
		{
			SVD_CUDA_CHECK(cudaMalloc(
				reinterpret_cast<void**>(&d_projected),
				sizeof(ComplexType) * static_cast<std::size_t>(pixels) * static_cast<std::size_t>(projected_cols)));
		}
		SVD_CUDA_CHECK(cudaEventCreate(&start));
		SVD_CUDA_CHECK(cudaEventCreate(&stop));

		SVD_CUSOLVER_CHECK(cusolverDnCheevd_bufferSize(
			cusolver_handle,
			CUSOLVER_EIG_MODE_VECTOR,
			CUBLAS_FILL_MODE_LOWER,
			n_frames,
			d_gram,
			n_frames,
			d_eigen_values,
			&lwork));
		if (lwork > 0)
		{
			SVD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_work), sizeof(ComplexType) * static_cast<std::size_t>(lwork)));
		}

		if (stats != nullptr)
		{
			stats->device_workspace_bytes =
				sizeof(ComplexType) * (gram_elements + static_cast<std::size_t>(std::max(lwork, 0))) +
				sizeof(float) * static_cast<std::size_t>(n_frames);
			stats->host_workspace_bytes = 0;
		}

		SVD_CUDA_CHECK(cudaEventRecord(start, stream));

		// Exact temporal SVD via covariance: X = pixels x frames, Gram = X^H X.
		SVD_CUBLAS_CHECK(cublasCgemm(
			cublas_handle,
			CUBLAS_OP_C,
			CUBLAS_OP_N,
			n_frames,
			n_frames,
			pixels,
			&alpha,
			d_a_base,
			pixels,
			d_a_base,
			pixels,
			&beta,
			d_gram,
			n_frames));

		SVD_CUSOLVER_CHECK(cusolverDnCheevd(
			cusolver_handle,
			CUSOLVER_EIG_MODE_VECTOR,
			CUBLAS_FILL_MODE_LOWER,
			n_frames,
			d_gram,
			n_frames,
			d_eigen_values,
			d_work,
			lwork,
			d_info));

		if (tail_filter)
		{
			const ComplexType* d_v_clutter =
				d_gram + static_cast<std::size_t>(n_frames - clutter_count) * static_cast<std::size_t>(n_frames);
			SVD_CUBLAS_CHECK(cublasCgemm(
				cublas_handle,
				CUBLAS_OP_N,
				CUBLAS_OP_N,
				pixels,
				clutter_count,
				n_frames,
				&alpha,
				d_a_base,
				pixels,
				d_v_clutter,
				n_frames,
				&beta,
				d_projected,
				pixels));
			SVD_CUBLAS_CHECK(cublasCgemm(
				cublas_handle,
				CUBLAS_OP_N,
				CUBLAS_OP_C,
				pixels,
				n_frames,
				clutter_count,
				&alpha,
				d_projected,
				pixels,
				d_v_clutter,
				n_frames,
				&beta,
				d_recon,
				pixels));
			SVD_CUBLAS_CHECK(cublasCgeam(
				cublas_handle,
				CUBLAS_OP_N,
				CUBLAS_OP_N,
				pixels,
				n_frames,
				&alpha,
				d_a_base,
				pixels,
				&minus_one,
				d_recon,
				pixels,
				d_output,
				pixels));
		}
		else
		{
			const int keep_start_ascending = n_frames - cutoff_end;
			const ComplexType* d_v_keep =
				d_gram + static_cast<std::size_t>(keep_start_ascending) * static_cast<std::size_t>(n_frames);
			SVD_CUBLAS_CHECK(cublasCgemm(
				cublas_handle,
				CUBLAS_OP_N,
				CUBLAS_OP_N,
				pixels,
				keep_count,
				n_frames,
				&alpha,
				d_a_base,
				pixels,
				d_v_keep,
				n_frames,
				&beta,
				d_projected,
				pixels));
			SVD_CUBLAS_CHECK(cublasCgemm(
				cublas_handle,
				CUBLAS_OP_N,
				CUBLAS_OP_C,
				pixels,
				n_frames,
				keep_count,
				&alpha,
				d_projected,
				pixels,
				d_v_keep,
				n_frames,
				&beta,
				d_output,
				pixels));
		}

		SVD_CUDA_CHECK(cudaEventRecord(stop, stream));
		SVD_CUDA_CHECK(cudaEventSynchronize(stop));

		int info = 0;
		SVD_CUDA_CHECK(cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));
		SVD_CUDA_CHECK(cudaStreamSynchronize(stream));
		ValidateSolverInfo(info);

		if (stats != nullptr)
		{
			SVD_CUDA_CHECK(cudaEventElapsedTime(&stats->elapsed_ms, start, stop));
			std::vector<float> eigen_values(static_cast<std::size_t>(n_frames));
			SVD_CUDA_CHECK(cudaMemcpy(
				eigen_values.data(),
				d_eigen_values,
				sizeof(float) * eigen_values.size(),
				cudaMemcpyDeviceToHost));
			stats->singular_values.resize(static_cast<std::size_t>(n_frames));
			for (int i = 0; i < n_frames; ++i)
			{
				const float lambda = std::max(eigen_values[static_cast<std::size_t>(n_frames - 1 - i)], 0.0f);
				stats->singular_values[static_cast<std::size_t>(i)] = std::sqrt(lambda);
			}
		}
	}
	catch (...)
	{
		cleanup();
		throw;
	}

	cleanup();
}

void EstimateSsmCutoffFromIqCache(
	const float2* d_iq_cache,
	int pixels,
	int n_frames,
	const SsmCutoffConfig& config,
	SsmCutoffResult* result,
	cudaStream_t stream)
{
	if (d_iq_cache == nullptr)
	{
		throw std::invalid_argument("d_iq_cache must not be null.");
	}
	if (result == nullptr)
	{
		throw std::invalid_argument("SSM result must not be null.");
	}

	ValidateSsmConfig(config, pixels, n_frames);
	*result = SsmCutoffResult{};

	const int cols = config.max_components;
	const int spatial_total = pixels * cols;
	const std::size_t gram_elements = static_cast<std::size_t>(n_frames) * static_cast<std::size_t>(n_frames);
	const std::size_t spatial_bytes = sizeof(ComplexType) * static_cast<std::size_t>(spatial_total);
	const std::size_t envelope_bytes = sizeof(float) * static_cast<std::size_t>(spatial_total);
	const std::size_t ssm_elements = static_cast<std::size_t>(cols) * static_cast<std::size_t>(cols);
	const ComplexType* const d_a_base = reinterpret_cast<const ComplexType*>(d_iq_cache);

	cublasHandle_t cublas_handle = nullptr;
	cusolverDnHandle_t cusolver_handle = nullptr;
	ComplexType* d_gram = nullptr;
	float* d_eigen_values = nullptr;
	ComplexType* d_spatial = nullptr;
	float* d_envelope = nullptr;
	float* d_column_sums = nullptr;
	float* d_norms = nullptr;
	float* d_ssm = nullptr;
	float* d_ones = nullptr;
	ComplexType* d_work = nullptr;
	int* d_info = nullptr;
	cudaEvent_t start = nullptr;
	cudaEvent_t stop = nullptr;
	int lwork = 0;

	const cuComplex alpha_c = make_cuComplex(1.0f, 0.0f);
	const cuComplex beta_c = make_cuComplex(0.0f, 0.0f);
	const float alpha_f = 1.0f;
	const float beta_f = 0.0f;

	auto cleanup = [&]() {
		if (stop != nullptr) { cudaEventDestroy(stop); }
		if (start != nullptr) { cudaEventDestroy(start); }
		if (d_info != nullptr) { cudaFree(d_info); }
		if (d_work != nullptr) { cudaFree(d_work); }
		if (d_ones != nullptr) { cudaFree(d_ones); }
		if (d_ssm != nullptr) { cudaFree(d_ssm); }
		if (d_norms != nullptr) { cudaFree(d_norms); }
		if (d_column_sums != nullptr) { cudaFree(d_column_sums); }
		if (d_envelope != nullptr) { cudaFree(d_envelope); }
		if (d_spatial != nullptr) { cudaFree(d_spatial); }
		if (d_eigen_values != nullptr) { cudaFree(d_eigen_values); }
		if (d_gram != nullptr) { cudaFree(d_gram); }
		if (cusolver_handle != nullptr) { cusolverDnDestroy(cusolver_handle); }
		if (cublas_handle != nullptr) { cublasDestroy(cublas_handle); }
	};

	try
	{
		SVD_CUBLAS_CHECK(cublasCreate(&cublas_handle));
		SVD_CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));
		SVD_CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));
		SVD_CUSOLVER_CHECK(cusolverDnSetStream(cusolver_handle, stream));

		SVD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gram), sizeof(ComplexType) * gram_elements));
		SVD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_eigen_values), sizeof(float) * static_cast<std::size_t>(n_frames)));
		SVD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_spatial), spatial_bytes));
		SVD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_envelope), envelope_bytes));
		SVD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_column_sums), sizeof(float) * static_cast<std::size_t>(cols)));
		SVD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_norms), sizeof(float) * static_cast<std::size_t>(cols)));
		SVD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ssm), sizeof(float) * ssm_elements));
		SVD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ones), sizeof(float) * static_cast<std::size_t>(pixels)));
		SVD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_info), sizeof(int)));
		SVD_CUDA_CHECK(cudaEventCreate(&start));
		SVD_CUDA_CHECK(cudaEventCreate(&stop));

		SVD_CUSOLVER_CHECK(cusolverDnCheevd_bufferSize(
			cusolver_handle,
			CUSOLVER_EIG_MODE_VECTOR,
			CUBLAS_FILL_MODE_LOWER,
			n_frames,
			d_gram,
			n_frames,
			d_eigen_values,
			&lwork));
		if (lwork > 0)
		{
			SVD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_work), sizeof(ComplexType) * static_cast<std::size_t>(lwork)));
		}

		result->max_components = cols;
		result->device_workspace_bytes =
			sizeof(ComplexType) * (gram_elements + static_cast<std::size_t>(std::max(lwork, 0))) +
			spatial_bytes +
			sizeof(float) * (static_cast<std::size_t>(n_frames) + 2 * static_cast<std::size_t>(cols) + ssm_elements + static_cast<std::size_t>(pixels)) +
			envelope_bytes;
		result->host_workspace_bytes = sizeof(float) * static_cast<std::size_t>(cols);

		const int threads = 256;
		const int fill_blocks = (pixels + threads - 1) / threads;
		const int spatial_blocks = (spatial_total + threads - 1) / threads;
		const int ssm_blocks = (static_cast<int>(ssm_elements) + threads - 1) / threads;

		SVD_CUDA_CHECK(cudaEventRecord(start, stream));

		SVD_CUBLAS_CHECK(cublasCgemm(
			cublas_handle,
			CUBLAS_OP_C,
			CUBLAS_OP_N,
			n_frames,
			n_frames,
			pixels,
			&alpha_c,
			d_a_base,
			pixels,
			d_a_base,
			pixels,
			&beta_c,
			d_gram,
			n_frames));

		SVD_CUSOLVER_CHECK(cusolverDnCheevd(
			cusolver_handle,
			CUSOLVER_EIG_MODE_VECTOR,
			CUBLAS_FILL_MODE_LOWER,
			n_frames,
			d_gram,
			n_frames,
			d_eigen_values,
			d_work,
			lwork,
			d_info));

		const ComplexType* d_v_top = d_gram + static_cast<std::size_t>(n_frames - cols) * static_cast<std::size_t>(n_frames);
		SVD_CUBLAS_CHECK(cublasCgemm(
			cublas_handle,
			CUBLAS_OP_N,
			CUBLAS_OP_N,
			pixels,
			cols,
			n_frames,
			&alpha_c,
			d_a_base,
			pixels,
			d_v_top,
			n_frames,
			&beta_c,
			d_spatial,
			pixels));

		CUDA_KERNEL_LAUNCH(
			KernelSpatialToEnvelopeDescending,
			spatial_blocks,
			threads,
			0,
			stream,
			d_spatial,
			d_eigen_values,
			d_envelope,
			pixels,
			n_frames,
			cols);
		SVD_CUDA_CHECK(cudaGetLastError());

		CUDA_KERNEL_LAUNCH(KernelFillFloat, fill_blocks, threads, 0, stream, d_ones, pixels, 1.0f);
		SVD_CUDA_CHECK(cudaGetLastError());

		SVD_CUBLAS_CHECK(cublasSgemv(
			cublas_handle,
			CUBLAS_OP_T,
			pixels,
			cols,
			&alpha_f,
			d_envelope,
			pixels,
			d_ones,
			1,
			&beta_f,
			d_column_sums,
			1));

		CUDA_KERNEL_LAUNCH(
			KernelCenterColumns,
			spatial_blocks,
			threads,
			0,
			stream,
			d_envelope,
			d_column_sums,
			pixels,
			cols);
		SVD_CUDA_CHECK(cudaGetLastError());

		std::vector<float> host_norms(static_cast<std::size_t>(cols), 0.0f);
		for (int col = 0; col < cols; ++col)
		{
			SVD_CUBLAS_CHECK(cublasSnrm2(
				cublas_handle,
				pixels,
				d_envelope + static_cast<std::size_t>(col) * static_cast<std::size_t>(pixels),
				1,
				&host_norms[static_cast<std::size_t>(col)]));
		}
		SVD_CUDA_CHECK(cudaMemcpyAsync(
			d_norms,
			host_norms.data(),
			sizeof(float) * host_norms.size(),
			cudaMemcpyHostToDevice,
			stream));

		CUDA_KERNEL_LAUNCH(
			KernelNormalizeColumns,
			spatial_blocks,
			threads,
			0,
			stream,
			d_envelope,
			d_norms,
			pixels,
			cols);
		SVD_CUDA_CHECK(cudaGetLastError());

		SVD_CUBLAS_CHECK(cublasSgemm(
			cublas_handle,
			CUBLAS_OP_T,
			CUBLAS_OP_N,
			cols,
			cols,
			pixels,
			&alpha_f,
			d_envelope,
			pixels,
			d_envelope,
			pixels,
			&beta_f,
			d_ssm,
			cols));

		CUDA_KERNEL_LAUNCH(KernelAbsInPlace, ssm_blocks, threads, 0, stream, d_ssm, static_cast<int>(ssm_elements));
		SVD_CUDA_CHECK(cudaGetLastError());

		SVD_CUDA_CHECK(cudaEventRecord(stop, stream));
		SVD_CUDA_CHECK(cudaEventSynchronize(stop));

		int info = 0;
		SVD_CUDA_CHECK(cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));
		SVD_CUDA_CHECK(cudaStreamSynchronize(stream));
		ValidateSolverInfo(info);

		SVD_CUDA_CHECK(cudaEventElapsedTime(&result->elapsed_ms, start, stop));

		std::vector<float> eigen_values(static_cast<std::size_t>(n_frames), 0.0f);
		SVD_CUDA_CHECK(cudaMemcpy(
			eigen_values.data(),
			d_eigen_values,
			sizeof(float) * eigen_values.size(),
			cudaMemcpyDeviceToHost));
		result->singular_values.resize(static_cast<std::size_t>(cols));
		for (int i = 0; i < cols; ++i)
		{
			const float lambda = std::max(eigen_values[static_cast<std::size_t>(n_frames - 1 - i)], 0.0f);
			result->singular_values[static_cast<std::size_t>(i)] = std::sqrt(lambda);
		}

		result->ssm_matrix.resize(ssm_elements);
		SVD_CUDA_CHECK(cudaMemcpy(
			result->ssm_matrix.data(),
			d_ssm,
			sizeof(float) * result->ssm_matrix.size(),
			cudaMemcpyDeviceToHost));

		FitTwoSquareModel(
			result->ssm_matrix,
			cols,
			config.min_tissue,
			config.min_flow,
			&result->tissue_component_count,
			&result->flow_component_end,
			&result->fit_score);

		result->cutoff_start = result->tissue_component_count + 1;
		result->cutoff_end = (cols == n_frames || result->flow_component_end < cols)
			? result->flow_component_end
			: n_frames;
	}
	catch (...)
	{
		cleanup();
		throw;
	}

	cleanup();
}

void RsvdTopKSubtractIqCache(
	const float2* d_iq_cache,
	int pixels,
	int n_frames,
	int remove_rank,
	int oversampling,
	int power_iterations,
	float2* d_filtered_iq_cache,
	RsvdFilterStats* stats,
	cudaStream_t stream)
{
	if (d_iq_cache == nullptr)
	{
		throw std::invalid_argument("d_iq_cache must not be null.");
	}
	if (d_filtered_iq_cache == nullptr)
	{
		throw std::invalid_argument("d_filtered_iq_cache must not be null.");
	}

	ValidateTopKConfig(pixels, n_frames, remove_rank, oversampling, power_iterations);

	if (stats != nullptr)
	{
		*stats = RsvdFilterStats{};
	}

	const int total = pixels * n_frames;
	const std::size_t matrix_bytes = sizeof(ComplexType) * static_cast<std::size_t>(total);
	const std::size_t singular_bytes = sizeof(SingularReal) * static_cast<std::size_t>(std::min(pixels, n_frames));
	const ComplexType* const d_a_base = reinterpret_cast<const ComplexType*>(d_iq_cache);
	ComplexType* const d_output = reinterpret_cast<ComplexType*>(d_filtered_iq_cache);

	cublasHandle_t cublas_handle = nullptr;
	cusolverDnHandle_t cusolver_handle = nullptr;
	cusolverDnParams_t params = nullptr;
	ComplexType* d_a_work = nullptr;
	SingularReal* d_singular_values = nullptr;
	ComplexType* d_dummy_u = nullptr;
	ComplexType* d_v = nullptr;
	ComplexType* d_projected = nullptr;
	ComplexType* d_recon = nullptr;
	int* d_info = nullptr;
	void* d_work = nullptr;
	void* h_work = nullptr;
	cudaEvent_t start = nullptr;
	cudaEvent_t stop = nullptr;

	const cuComplex alpha = make_cuComplex(1.0f, 0.0f);
	const cuComplex beta = make_cuComplex(0.0f, 0.0f);
	const cuComplex minus_one = make_cuComplex(-1.0f, 0.0f);

	auto cleanup = [&]() {
		if (stop != nullptr) { cudaEventDestroy(stop); }
		if (start != nullptr) { cudaEventDestroy(start); }
		if (h_work != nullptr) { std::free(h_work); }
		if (d_work != nullptr) { cudaFree(d_work); }
		if (d_info != nullptr) { cudaFree(d_info); }
		if (d_recon != nullptr) { cudaFree(d_recon); }
		if (d_projected != nullptr) { cudaFree(d_projected); }
		if (d_v != nullptr) { cudaFree(d_v); }
		if (d_dummy_u != nullptr) { cudaFree(d_dummy_u); }
		if (d_singular_values != nullptr) { cudaFree(d_singular_values); }
		if (d_a_work != nullptr) { cudaFree(d_a_work); }
		if (params != nullptr) { cusolverDnDestroyParams(params); }
		if (cusolver_handle != nullptr) { cusolverDnDestroy(cusolver_handle); }
		if (cublas_handle != nullptr) { cublasDestroy(cublas_handle); }
	};

	try
	{
		SVD_CUBLAS_CHECK(cublasCreate(&cublas_handle));
		SVD_CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));
		SVD_CUSOLVER_CHECK(cusolverDnCreateParams(&params));
		SVD_CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));
		SVD_CUSOLVER_CHECK(cusolverDnSetStream(cusolver_handle, stream));

		SVD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_a_work), matrix_bytes));
		SVD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_singular_values), singular_bytes));
		SVD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_dummy_u), sizeof(ComplexType)));
		SVD_CUDA_CHECK(cudaMalloc(
			reinterpret_cast<void**>(&d_v),
			sizeof(ComplexType) * static_cast<std::size_t>(n_frames) * static_cast<std::size_t>(n_frames)));
		SVD_CUDA_CHECK(cudaMalloc(
			reinterpret_cast<void**>(&d_projected),
			sizeof(ComplexType) * static_cast<std::size_t>(pixels) * static_cast<std::size_t>(remove_rank)));
		SVD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_recon), matrix_bytes));
		SVD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_info), sizeof(int)));
		SVD_CUDA_CHECK(cudaEventCreate(&start));
		SVD_CUDA_CHECK(cudaEventCreate(&stop));

		std::size_t device_workspace_bytes = 0;
		std::size_t host_workspace_bytes = 0;
		SVD_CUSOLVER_CHECK(cusolverDnXgesvdr_bufferSize(
			cusolver_handle,
			params,
			'N',
			'S',
			static_cast<int64_t>(pixels),
			static_cast<int64_t>(n_frames),
			static_cast<int64_t>(remove_rank),
			static_cast<int64_t>(oversampling),
			static_cast<int64_t>(power_iterations),
			CUDA_C_32F,
			d_a_work,
			static_cast<int64_t>(pixels),
			CUDA_R_32F,
			d_singular_values,
			CUDA_C_32F,
			d_dummy_u,
			static_cast<int64_t>(pixels),
			CUDA_C_32F,
			d_v,
			static_cast<int64_t>(n_frames),
			CUDA_C_32F,
			&device_workspace_bytes,
			&host_workspace_bytes));

		if (device_workspace_bytes > 0)
		{
			SVD_CUDA_CHECK(cudaMalloc(&d_work, device_workspace_bytes));
		}
		if (host_workspace_bytes > 0)
		{
			h_work = std::malloc(host_workspace_bytes);
			if (h_work == nullptr)
			{
				throw std::bad_alloc();
			}
		}

		if (stats != nullptr)
		{
			stats->device_workspace_bytes = device_workspace_bytes;
			stats->host_workspace_bytes = host_workspace_bytes;
		}

		SVD_CUDA_CHECK(cudaEventRecord(start, stream));
		SVD_CUDA_CHECK(cudaMemcpyAsync(d_a_work, d_a_base, matrix_bytes, cudaMemcpyDeviceToDevice, stream));
		SVD_CUSOLVER_CHECK(cusolverDnXgesvdr(
			cusolver_handle,
			params,
			'N',
			'S',
			static_cast<int64_t>(pixels),
			static_cast<int64_t>(n_frames),
			static_cast<int64_t>(remove_rank),
			static_cast<int64_t>(oversampling),
			static_cast<int64_t>(power_iterations),
			CUDA_C_32F,
			d_a_work,
			static_cast<int64_t>(pixels),
			CUDA_R_32F,
			d_singular_values,
			CUDA_C_32F,
			d_dummy_u,
			static_cast<int64_t>(pixels),
			CUDA_C_32F,
			d_v,
			static_cast<int64_t>(n_frames),
			CUDA_C_32F,
			d_work,
			device_workspace_bytes,
			h_work,
			host_workspace_bytes,
			d_info));

		// Remove the low-rank clutter estimate using the right singular vectors:
		// X_filtered = X - X * V_k * V_k^H.
		SVD_CUBLAS_CHECK(cublasCgemm(
			cublas_handle,
			CUBLAS_OP_N,
			CUBLAS_OP_N,
			pixels,
			remove_rank,
			n_frames,
			&alpha,
			d_a_base,
			pixels,
			d_v,
			n_frames,
			&beta,
			d_projected,
			pixels));
		SVD_CUBLAS_CHECK(cublasCgemm(
			cublas_handle,
			CUBLAS_OP_N,
			CUBLAS_OP_C,
			pixels,
			n_frames,
			remove_rank,
			&alpha,
			d_projected,
			pixels,
			d_v,
			n_frames,
			&beta,
			d_recon,
			pixels));
		SVD_CUBLAS_CHECK(cublasCgeam(
			cublas_handle,
			CUBLAS_OP_N,
			CUBLAS_OP_N,
			pixels,
			n_frames,
			&alpha,
			d_a_base,
			pixels,
			&minus_one,
			d_recon,
			pixels,
			d_output,
			pixels));

		SVD_CUDA_CHECK(cudaEventRecord(stop, stream));
		SVD_CUDA_CHECK(cudaEventSynchronize(stop));

		int info = 0;
		SVD_CUDA_CHECK(cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));
		SVD_CUDA_CHECK(cudaStreamSynchronize(stream));
		ValidateSolverInfo(info);

		if (stats != nullptr)
		{
			SVD_CUDA_CHECK(cudaEventElapsedTime(&stats->elapsed_ms, start, stop));
			stats->singular_values.resize(static_cast<std::size_t>(remove_rank));
			SVD_CUDA_CHECK(cudaMemcpy(
				stats->singular_values.data(),
				d_singular_values,
				sizeof(float) * stats->singular_values.size(),
				cudaMemcpyDeviceToHost));
		}
	}
	catch (...)
	{
		cleanup();
		throw;
	}

	cleanup();
}

void PowerDopplerFromIqCache(
	const float2* d_iq_cache,
	int pixels,
	int n_frames,
	float* d_power,
	cudaStream_t stream)
{
	if (d_iq_cache == nullptr)
	{
		throw std::invalid_argument("d_iq_cache must not be null.");
	}
	if (d_power == nullptr)
	{
		throw std::invalid_argument("d_power must not be null.");
	}
	if (pixels <= 0 || n_frames <= 0)
	{
		throw std::invalid_argument("pixels and n_frames must be positive.");
	}

	const int threads = 256;
	const int blocks = (pixels + threads - 1) / threads;
	CUDA_KERNEL_LAUNCH(
		Kernel_PowerDopplerFromIqCache,
		blocks,
		threads,
		0,
		stream,
		d_iq_cache,
		pixels,
		n_frames,
		d_power);
	SVD_CUDA_CHECK(cudaGetLastError());
}
