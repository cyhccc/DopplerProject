#include "data_preprocess_gpu_v2.cuh"

#include "config.cuh"
#include "cuda_intellisense.cuh"

#include <cuComplex.h>
#include <cufft.h>
#include <cstdint>
#include <stdexcept>

namespace
{
	constexpr int kSampleTile = 32;
	constexpr int kFrameTile = 128;
	constexpr int kFrameRows = 8;

	__global__ void ApplyHilbertHalfSpectrumKernel(
		cufftComplex* __restrict__ d_half,
		int samples,
		int half_bins)
	{
		const std::size_t batch = static_cast<std::size_t>(blockIdx.x);
		const std::size_t half_base = batch * static_cast<std::size_t>(half_bins);
		for (int k = static_cast<int>(threadIdx.x); k < half_bins; k += static_cast<int>(blockDim.x))
		{
			const bool edge = (k == 0) || (((samples & 1) == 0) && (k == samples / 2));
			cufftComplex output = make_cuFloatComplex(0.0f, 0.0f);
			if (!edge)
			{
				const cufftComplex value = d_half[half_base + static_cast<std::size_t>(k)];
				output = make_cuFloatComplex(value.y, -value.x);
			}
			d_half[half_base + static_cast<std::size_t>(k)] = output;
		}
	}

	__global__ void NormalizeHilbertFrameMajorKernel(
		float* __restrict__ d_hilbert,
		int samples,
		int frames,
		float scale)
	{
		const int sample = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
		const int seq_channel = static_cast<int>(blockIdx.y);
		if (sample >= samples)
		{
			return;
		}

		const std::size_t seq_channels = static_cast<std::size_t>(gridDim.y);
		for (int frame = static_cast<int>(threadIdx.y); frame < frames; frame += static_cast<int>(blockDim.y))
		{
			const std::size_t idx =
				(static_cast<std::size_t>(frame) * seq_channels + seq_channel) *
				static_cast<std::size_t>(samples) + sample;
			d_hilbert[idx] *= scale;
		}
	}

	__global__ void StoreIqBeamformerLayoutTiledKernel(
		const float* __restrict__ d_rf,
		const float* __restrict__ d_hilbert,
		float2* __restrict__ d_iq,
		int samples,
		int padded_samples,
		int channels,
		int frames,
		float inv_padded_samples)
	{
		__shared__ float2 tile[kFrameTile][kSampleTile + 1];

		const int sample_base = static_cast<int>(blockIdx.x) * kSampleTile;
		const int frame_base = static_cast<int>(blockIdx.y) * kFrameTile;
		const int seq_channel = static_cast<int>(blockIdx.z);
		const std::size_t seq_channels = static_cast<std::size_t>(gridDim.z);
		const int seq_idx = seq_channel / channels;
		const int channel = seq_channel - seq_idx * channels;

		for (int j = 0; j < kFrameTile; j += kFrameRows)
		{
			const int sample_in = sample_base + threadIdx.x;
			const int frame_in = frame_base + threadIdx.y + j;
			if (sample_in < samples && frame_in < frames)
			{
				const std::size_t input_trace =
					static_cast<std::size_t>(frame_in) * seq_channels + static_cast<std::size_t>(seq_channel);
				const std::size_t input_idx = input_trace * static_cast<std::size_t>(samples) + sample_in;
				const std::size_t padded_idx = input_trace * static_cast<std::size_t>(padded_samples) + sample_in;
				tile[threadIdx.y + j][threadIdx.x] =
					make_float2(d_rf[input_idx], d_hilbert[padded_idx] * inv_padded_samples);
			}
		}
		__syncthreads();

		for (int sample_j = 0; sample_j < kSampleTile; sample_j += kFrameRows)
		{
			for (int frame_j = 0; frame_j < kFrameTile; frame_j += kSampleTile)
			{
				const int sample_out = sample_base + threadIdx.y + sample_j;
				const int frame_out = frame_base + threadIdx.x + frame_j;
				if (sample_out < samples && frame_out < frames)
				{
					const std::size_t output_idx =
						((static_cast<std::size_t>(seq_idx) * channels + channel) * samples + sample_out) *
						static_cast<std::size_t>(frames) + frame_out;
					d_iq[output_idx] = tile[threadIdx.x + frame_j][threadIdx.y + sample_j];
				}
			}
		}
	}

	cufftHandle& AsPlan(void* plan_storage)
	{
		return *reinterpret_cast<cufftHandle*>(plan_storage);
	}
}

FieldIIRfToIqPreprocessor::FieldIIRfToIqPreprocessor(int samples, int channels, int seq, int frames)
	: samples_(samples),
	channels_(channels),
	seq_(seq),
	frames_(frames)
{
	if (samples_ <= 0 || channels_ <= 0 || seq_ <= 0 || frames_ <= 0)
	{
		throw std::invalid_argument("RF preprocessor dimensions must be positive.");
	}

	padded_samples_ = samples_;
	const std::size_t padded_total =
		static_cast<std::size_t>(padded_samples_) * channels_ * seq_ * frames_;
	const int half_bins = padded_samples_ / 2 + 1;
	const std::size_t half_total = static_cast<std::size_t>(half_bins) * channels_ * seq_ * frames_;
	CUDA_CHECK(cudaMalloc(&d_half_spectrum_, sizeof(cufftComplex) * half_total));
	CUDA_CHECK(cudaMalloc(&d_padded_time_, sizeof(float) * padded_total));

	r2c_plan_ = new cufftHandle{};
	inverse_plan_ = new cufftHandle{};
	int n[] = { padded_samples_ };
	const int batch = channels_ * seq_ * frames_;
	CUFFT_CHECK(cufftPlanMany(
		&AsPlan(r2c_plan_),
		1,
		n,
		nullptr,
		1,
		padded_samples_,
		nullptr,
		1,
		half_bins,
		CUFFT_R2C,
		batch));
	CUFFT_CHECK(cufftPlanMany(
		&AsPlan(inverse_plan_),
		1,
		n,
		nullptr,
		1,
		half_bins,
		nullptr,
		1,
		padded_samples_,
		CUFFT_C2R,
		batch));
}

FieldIIRfToIqPreprocessor::~FieldIIRfToIqPreprocessor()
{
	if (inverse_plan_ != nullptr)
	{
		cufftDestroy(AsPlan(inverse_plan_));
		delete reinterpret_cast<cufftHandle*>(inverse_plan_);
		inverse_plan_ = nullptr;
	}
	if (r2c_plan_ != nullptr)
	{
		cufftDestroy(AsPlan(r2c_plan_));
		delete reinterpret_cast<cufftHandle*>(r2c_plan_);
		r2c_plan_ = nullptr;
	}
	if (d_padded_time_ != nullptr)
	{
		cudaFree(d_padded_time_);
		d_padded_time_ = nullptr;
	}
	if (d_half_spectrum_ != nullptr)
	{
		cudaFree(d_half_spectrum_);
		d_half_spectrum_ = nullptr;
	}
}

void FieldIIRfToIqPreprocessor::RfToIq(const float* d_rf, float2* d_iq, cudaStream_t stream)
{
	if (d_rf == nullptr || d_iq == nullptr)
	{
		throw std::invalid_argument("RF preprocessor input/output must not be null.");
	}

	CUFFT_CHECK(cufftSetStream(AsPlan(r2c_plan_), stream));
	CUFFT_CHECK(cufftSetStream(AsPlan(inverse_plan_), stream));

	cufftComplex* const d_half = reinterpret_cast<cufftComplex*>(d_half_spectrum_);
	float* const d_padded = reinterpret_cast<float*>(d_padded_time_);
	const int half_bins = padded_samples_ / 2 + 1;

	const int batch = channels_ * seq_ * frames_;
	CUFFT_CHECK(cufftExecR2C(AsPlan(r2c_plan_), const_cast<float*>(d_rf), d_half));
	(void)cudaGetLastError();

	const int spectrum_threads = 128;
	const int spectrum_blocks = batch;
	ApplyHilbertHalfSpectrumKernel<<<spectrum_blocks, spectrum_threads, 0, stream>>>(
		d_half,
		padded_samples_,
		half_bins);
	CUDA_CHECK(cudaGetLastError());

	CUFFT_CHECK(cufftExecC2R(AsPlan(inverse_plan_), d_half, d_padded));
	(void)cudaGetLastError();

	StoreFieldIIRfHilbertBeamformerLayout(
		d_rf,
		d_padded,
		d_iq,
		samples_,
		padded_samples_,
		channels_,
		seq_,
		frames_,
		1.0f / static_cast<float>(padded_samples_),
		stream);
}

void FieldIIRfToIqPreprocessor::RfToHilbertFrameMajor(const float* d_rf, cudaStream_t stream)
{
	if (d_rf == nullptr)
	{
		throw std::invalid_argument("RF preprocessor input must not be null.");
	}

	CUFFT_CHECK(cufftSetStream(AsPlan(r2c_plan_), stream));
	CUFFT_CHECK(cufftSetStream(AsPlan(inverse_plan_), stream));

	cufftComplex* const d_half = reinterpret_cast<cufftComplex*>(d_half_spectrum_);
	float* const d_hilbert = reinterpret_cast<float*>(d_padded_time_);
	const int half_bins = padded_samples_ / 2 + 1;
	const int batch = channels_ * seq_ * frames_;

	CUFFT_CHECK(cufftExecR2C(AsPlan(r2c_plan_), const_cast<float*>(d_rf), d_half));
	(void)cudaGetLastError();

	const int spectrum_threads = 128;
	ApplyHilbertHalfSpectrumKernel<<<batch, spectrum_threads, 0, stream>>>(
		d_half,
		padded_samples_,
		half_bins);
	CUDA_CHECK(cudaGetLastError());

	CUFFT_CHECK(cufftExecC2R(AsPlan(inverse_plan_), d_half, d_hilbert));
	(void)cudaGetLastError();

	const dim3 normalize_threads(kSampleTile, kFrameRows, 1);
	const dim3 normalize_blocks(
		(samples_ + kSampleTile - 1) / kSampleTile,
		seq_ * channels_,
		1);
	NormalizeHilbertFrameMajorKernel<<<normalize_blocks, normalize_threads, 0, stream>>>(
		d_hilbert,
		samples_,
		frames_,
		1.0f / static_cast<float>(padded_samples_));
	CUDA_CHECK(cudaGetLastError());
}

const float* FieldIIRfToIqPreprocessor::HilbertFrameMajor() const
{
	return reinterpret_cast<const float*>(d_padded_time_);
}

std::size_t FieldIIRfToIqPreprocessor::WorkspaceBytes() const
{
	const int half_bins = padded_samples_ / 2 + 1;
	return sizeof(cufftComplex) *
		static_cast<std::size_t>(channels_) *
		static_cast<std::size_t>(seq_) *
		static_cast<std::size_t>(frames_) *
		static_cast<std::size_t>(half_bins) +
		sizeof(float) *
		static_cast<std::size_t>(channels_) *
		static_cast<std::size_t>(seq_) *
		static_cast<std::size_t>(frames_) *
		static_cast<std::size_t>(padded_samples_);
}

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
	cudaStream_t stream)
{
	if (d_rf == nullptr || d_hilbert_frame_major == nullptr || d_iq == nullptr)
	{
		throw std::invalid_argument("Layout conversion input/output must not be null.");
	}
	if (samples <= 0 || channels <= 0 || seq <= 0 || frames <= 0)
	{
		throw std::invalid_argument("Layout conversion dimensions must be positive.");
	}
	if (padded_samples < samples)
	{
		throw std::invalid_argument("Layout conversion padded samples must cover samples.");
	}

	const dim3 transpose_threads(kSampleTile, kFrameRows, 1);
	const dim3 transpose_blocks(
		(samples + kSampleTile - 1) / kSampleTile,
		(frames + kFrameTile - 1) / kFrameTile,
		seq * channels);
	StoreIqBeamformerLayoutTiledKernel<<<transpose_blocks, transpose_threads, 0, stream>>>(
		d_rf,
		d_hilbert_frame_major,
		d_iq,
		samples,
		padded_samples,
		channels,
		frames,
		hilbert_scale);
	CUDA_CHECK(cudaGetLastError());
}
