#include "beamformer.cuh"

#include "config.cuh"
#include "cuda_intellisense.cuh"

#include <math_constants.h>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

extern PARAMs pc;

namespace
{
	constexpr int kPowerFrameThreads = 32;
	constexpr int kPowerVoxelTile = 8;

	__device__ __forceinline__ float2 SampleLinearGlobalFrameFastFma(
		const float2* sig,
		int i0,
		float frac,
		int n_frames,
		int frame_idx)
	{
		const size_t idx0 = static_cast<size_t>(i0) * n_frames + frame_idx;
		const size_t idx1 = idx0 + n_frames;
		const float2 s0 = sig[idx0];
		const float2 s1 = sig[idx1];

		float2 out;
		out.x = fmaf(frac, s1.x - s0.x, s0.x);
		out.y = fmaf(frac, s1.y - s0.y, s0.y);
		return out;
	}

	__device__ __forceinline__ float ComputeTxDelaySamples(
		float pixel_x,
		float pixel_z,
		float tx_dir_x,
		float tx_dir_z,
		float delay_bias,
		const PARAMs& params)
	{
		float tx_origin_distance = 0.0f;
		if (params.tx_reference_aperture_edge)
		{
			const float half_aperture = 0.5f * static_cast<float>(params.channels - 1) * params.pitch;
			const float edge_x = (tx_dir_x >= 0.0f) ? -half_aperture : half_aperture;
			tx_origin_distance = edge_x * tx_dir_x;
		}
		return (pixel_z * tx_dir_z + pixel_x * tx_dir_x - tx_origin_distance) *
			params.fs_div_sound_speed - delay_bias;
	}

	__device__ __forceinline__ void ComputeActiveChannelBounds(
		float pixel_x,
		float pixel_z,
		int channels,
		const PARAMs& params,
		int& ch_min,
		int& ch_max)
	{
		const float elem_x_origin = -0.5f * static_cast<float>(channels - 1) * params.pitch;
		const float apod_depth = fmaxf(pixel_z, params.F_number * params.minimum_aperture);
		const float aperture_radius = apod_depth / (2.0f * params.F_number);
		const float lo = (pixel_x - aperture_radius - elem_x_origin) / params.pitch;
		const float hi = (pixel_x + aperture_radius - elem_x_origin) / params.pitch;
		ch_min = max(0, static_cast<int>(ceilf(lo)));
		ch_max = min(channels - 1, static_cast<int>(floorf(hi)));
	}

	template<bool ApplyPhase>
	__device__ __forceinline__ float2 AccumulateChannelOuterIqDynamic(
		const float2* __restrict__ d_sig,
		const float* __restrict__ voxel_weight,
		const float* __restrict__ voxel_rx_delay,
		const float* __restrict__ voxel_tx_delay,
		int channels,
		int ch_min,
		int ch_max,
		int datalen,
		int seq,
		int n_frames,
		int frame_idx,
		size_t seq_stride,
		float phase_scale)
	{
		float2 coherent_sum = make_float2(0.0f, 0.0f);
		for (int ch = ch_min; ch <= ch_max; ++ch)
		{
			const float weight = voxel_weight[ch];
			if (weight <= 0.0f)
			{
				continue;
			}

			const float rx_delay = voxel_rx_delay[ch];
			const size_t channel_offset = static_cast<size_t>(ch) * datalen * n_frames;
			for (int s = 0; s < seq; ++s)
			{
				const float delay_value = voxel_tx_delay[s] + rx_delay;
				if (delay_value < 0.0f || delay_value >= datalen - 1.0f)
				{
					continue;
				}

				const int i0 = static_cast<int>(delay_value);
				const float frac = delay_value - static_cast<float>(i0);
				const float2 sample = SampleLinearGlobalFrameFastFma(
					d_sig + static_cast<size_t>(s) * seq_stride + channel_offset,
					i0,
					frac,
					n_frames,
					frame_idx);

				if constexpr (ApplyPhase)
				{
					const float phi = phase_scale * delay_value;
					float sin_phi, cos_phi;
					__sincosf(phi, &sin_phi, &cos_phi);
					coherent_sum.x = fmaf(weight * sample.x, cos_phi, fmaf(-weight * sample.y, sin_phi, coherent_sum.x));
					coherent_sum.y = fmaf(weight * sample.y, cos_phi, fmaf(weight * sample.x, sin_phi, coherent_sum.y));
				}
				else
				{
					coherent_sum.x += weight * sample.x;
					coherent_sum.y += weight * sample.y;
				}
			}
		}
		return coherent_sum;
	}

	template<bool ApplyPhase>
	__device__ __forceinline__ void WriteIqCacheFramesDynamic(
		const float2* __restrict__ d_sig,
		float2* d_iq_cache,
		int pixel_idx,
		int sum_pixel,
		const float* __restrict__ voxel_weight,
		const float* __restrict__ voxel_rx_delay,
		const float* __restrict__ voxel_tx_delay,
		int frame_lane,
		int channels,
		int ch_min,
		int ch_max,
		int datalen,
		int seq,
		int n_frames,
		size_t seq_stride,
		float phase_scale)
	{
		for (int frame_idx = frame_lane; frame_idx < n_frames; frame_idx += blockDim.x)
		{
			const float2 coherent_sum = AccumulateChannelOuterIqDynamic<ApplyPhase>(
				d_sig,
				voxel_weight,
				voxel_rx_delay,
				voxel_tx_delay,
				channels,
				ch_min,
				ch_max,
				datalen,
				seq,
				n_frames,
				frame_idx,
				seq_stride,
				phase_scale);
			d_iq_cache[static_cast<size_t>(frame_idx) * sum_pixel + pixel_idx] = coherent_sum;
		}
	}

}

BEAMFORMER::BEAMFORMER() = default;

BEAMFORMER::~BEAMFORMER() = default;

void BEAMFORMER::BeamformIQCacheSharedRectChannelOuterCore(
	const float2* d_sig,
	int n_frames,
	float2* d_iq_cache)
{
	if (n_frames <= 0)
	{
		throw std::runtime_error("n_frames must be positive for IQ-cache beamforming.");
	}
	if (d_iq_cache == nullptr)
	{
		throw std::runtime_error("d_iq_cache must not be null.");
	}

	const int sum_pixel = pc.img_nx * pc.img_nz;
	const dim3 block(kPowerFrameThreads, kPowerVoxelTile);
	const dim3 grid((sum_pixel + kPowerVoxelTile - 1) / kPowerVoxelTile);

	const size_t rx_delay_bytes = static_cast<size_t>(kPowerVoxelTile) * pc.channels * sizeof(float);
	const size_t tx_delay_bytes = static_cast<size_t>(kPowerVoxelTile) * pc.seq * sizeof(float);
	const size_t weight_bytes = static_cast<size_t>(kPowerVoxelTile) * pc.channels * sizeof(float);
	const size_t shared_bytes = rx_delay_bytes + tx_delay_bytes + weight_bytes;

	CUDA_KERNEL_LAUNCH(
		Kernel_Linear_Array_CPWC_IQCache_Row32x8_SharedRectChannelOuter,
		grid,
		block,
		shared_bytes,
		0,
		d_sig,
		d_iq_cache,
		sum_pixel,
		pc.channels,
		pc.datalen,
		pc.seq,
		n_frames,
		pc
	);
	CHECK_KERNEL();
}

__global__ void Kernel_Linear_Array_CPWC_IQCache_Row32x8_SharedRectChannelOuter(
	const float2* __restrict__ d_sig,
	float2* d_iq_cache,
	int sum_pixel,
	int channels,
	int datalen,
	int seq,
	int n_frames,
	PARAMs pc
)
{
	const int frame_lane = threadIdx.x;
	const int voxel_lane = threadIdx.y;
	const int pixel_idx = blockIdx.x * blockDim.y + voxel_lane;
	const bool pixel_valid = pixel_idx < sum_pixel;
	const bool apply_phase_rotation = pc.f0 > 0.0f;
	const float phase_scale = 2.0f * CUDART_PI_F * pc.f0 / pc.fs;
	const float elem_x_origin = -0.5f * static_cast<float>(channels - 1) * pc.pitch;
	const size_t seq_stride = static_cast<size_t>(channels) * datalen * n_frames;
	extern __shared__ float shared_workspace[];
	char* const shared_bytes = reinterpret_cast<char*>(shared_workspace);
	const size_t rx_delay_bytes = static_cast<size_t>(blockDim.y) * channels * sizeof(float);
	const size_t tx_delay_bytes = static_cast<size_t>(blockDim.y) * seq * sizeof(float);
	float* s_rx_delay = reinterpret_cast<float*>(shared_bytes);
	float* s_tx_delay = reinterpret_cast<float*>(shared_bytes + rx_delay_bytes);
	float* s_weight = reinterpret_cast<float*>(shared_bytes + rx_delay_bytes + tx_delay_bytes);

	float static_pixel_x = 0.0f;
	float static_pixel_z = 0.0f;
	if (pixel_valid && frame_lane == 0)
	{
		const int pixel_x_idx = pixel_idx % pc.img_nx;
		const int pixel_z_idx = pixel_idx / pc.img_nx;
		static_pixel_x = pc.x_start + static_cast<float>(pixel_x_idx) * pc.pixel_dx;
		static_pixel_z = pc.z_start + static_cast<float>(pixel_z_idx) * pc.pixel_dz;
	}

	static_pixel_x = __shfl_sync(0xffffffff, static_pixel_x, 0);
	static_pixel_z = __shfl_sync(0xffffffff, static_pixel_z, 0);

	int ch_min = 0;
	int ch_max = -1;
	if (pixel_valid)
	{
		ComputeActiveChannelBounds(static_pixel_x, static_pixel_z, channels, pc, ch_min, ch_max);
	}

	if (pixel_valid)
	{
		const float apod_depth = fmaxf(static_pixel_z, pc.F_number * pc.minimum_aperture);
		const float aperture_radius = apod_depth / (2.0f * pc.F_number);
		for (int ch = ch_min + frame_lane; ch <= ch_max; ch += blockDim.x)
		{
			const float elem_x = elem_x_origin + static_cast<float>(ch) * pc.pitch;
			const float rx_dx = static_pixel_x - elem_x;
			const float weight = ((aperture_radius > 0.0f) && (fabsf(rx_dx) <= aperture_radius))
				? 1.0f
				: 0.0f;
			const size_t shared_idx = static_cast<size_t>(voxel_lane) * channels + ch;
			s_weight[shared_idx] = weight;
			s_rx_delay[shared_idx] = (weight > 0.0f)
				? sqrtf(rx_dx * rx_dx + static_pixel_z * static_pixel_z) * pc.fs_div_sound_speed
				: 0.0f;
		}

		for (int s = frame_lane; s < seq; s += blockDim.x)
		{
			const float angle = pc.init_angle + static_cast<float>(s) * pc.angle_step;
			float tx_x = 0.0f;
			float tx_z = 0.0f;
			__sincosf(angle, &tx_x, &tx_z);
			s_tx_delay[static_cast<size_t>(voxel_lane) * seq + s] =
				ComputeTxDelaySamples(static_pixel_x, static_pixel_z, tx_x, tx_z, pc.tx_delay_offset_samples, pc);
		}
	}
	__syncthreads();

	if (!pixel_valid)
	{
		return;
	}

	const float* const voxel_weight = s_weight + static_cast<size_t>(voxel_lane) * channels;
	const float* const voxel_rx_delay = s_rx_delay + static_cast<size_t>(voxel_lane) * channels;
	const float* const voxel_tx_delay = s_tx_delay + static_cast<size_t>(voxel_lane) * seq;
	if (apply_phase_rotation)
	{
		WriteIqCacheFramesDynamic<true>(
			d_sig,
			d_iq_cache,
			pixel_idx,
			sum_pixel,
			voxel_weight,
			voxel_rx_delay,
			voxel_tx_delay,
			frame_lane,
			channels,
			ch_min,
			ch_max,
			datalen,
			seq,
			n_frames,
			seq_stride,
			phase_scale);
	}
	else
	{
		WriteIqCacheFramesDynamic<false>(
			d_sig,
			d_iq_cache,
			pixel_idx,
			sum_pixel,
			voxel_weight,
			voxel_rx_delay,
			voxel_tx_delay,
			frame_lane,
			channels,
			ch_min,
			ch_max,
			datalen,
			seq,
			n_frames,
			seq_stride,
			phase_scale);
	}
}
