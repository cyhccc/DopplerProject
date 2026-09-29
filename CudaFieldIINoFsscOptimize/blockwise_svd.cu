#include "blockwise_svd.cuh"

#include "config.cuh"

#include <stdexcept>

namespace
{
	__global__ void GatherIqBlockFrameMajorKernel(
		const float2* __restrict__ full,
		float2* __restrict__ block,
		int img_nx,
		int img_nz,
		int frames,
		int x0,
		int z0,
		int block_w,
		int block_h)
	{
		const int block_pixels = block_w * block_h;
		const int total = block_pixels * frames;
		const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
		if (idx >= total)
		{
			return;
		}

		const int frame = idx / block_pixels;
		const int local = idx - frame * block_pixels;
		const int lx = local % block_w;
		const int lz = local / block_w;
		const int gx = x0 + lx;
		const int gz = z0 + lz;
		if (gx >= img_nx || gz >= img_nz)
		{
			block[idx] = make_float2(0.0f, 0.0f);
			return;
		}
		const int full_pixels = img_nx * img_nz;
		const int global_pixel = gz * img_nx + gx;
		block[idx] = full[static_cast<size_t>(frame) * full_pixels + global_pixel];
	}

	__global__ void ScatterIqBlockFrameMajorKernel(
		const float2* __restrict__ block,
		float2* __restrict__ full,
		int img_nx,
		int img_nz,
		int frames,
		int x0,
		int z0,
		int block_w,
		int block_h)
	{
		const int block_pixels = block_w * block_h;
		const int total = block_pixels * frames;
		const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
		if (idx >= total)
		{
			return;
		}

		const int frame = idx / block_pixels;
		const int local = idx - frame * block_pixels;
		const int lx = local % block_w;
		const int lz = local / block_w;
		const int gx = x0 + lx;
		const int gz = z0 + lz;
		if (gx >= img_nx || gz >= img_nz)
		{
			return;
		}
		const int full_pixels = img_nx * img_nz;
		const int global_pixel = gz * img_nx + gx;
		full[static_cast<size_t>(frame) * full_pixels + global_pixel] = block[idx];
	}

	void ValidateBlockArgs(
		const float2* source,
		float2* destination,
		int img_nx,
		int img_nz,
		int frames,
		int x0,
		int z0,
		int block_w,
		int block_h)
	{
		if (source == nullptr || destination == nullptr)
		{
			throw std::invalid_argument("Block-wise SVD gather/scatter pointer must not be null.");
		}
		if (img_nx <= 0 || img_nz <= 0 || frames <= 0 || block_w <= 0 || block_h <= 0)
		{
			throw std::invalid_argument("Block-wise SVD gather/scatter dimensions must be positive.");
		}
		if (x0 < 0 || z0 < 0 || x0 >= img_nx || z0 >= img_nz)
		{
			throw std::invalid_argument("Block-wise SVD block origin is outside the image.");
		}
	}
}

void GatherIqBlockFrameMajor(
	const float2* d_full,
	float2* d_block,
	int img_nx,
	int img_nz,
	int frames,
	int x0,
	int z0,
	int block_w,
	int block_h,
	cudaStream_t stream)
{
	ValidateBlockArgs(d_full, d_block, img_nx, img_nz, frames, x0, z0, block_w, block_h);
	const int total = block_w * block_h * frames;
	const int threads = 256;
	const int blocks = (total + threads - 1) / threads;
	GatherIqBlockFrameMajorKernel<<<blocks, threads, 0, stream>>>(
		d_full,
		d_block,
		img_nx,
		img_nz,
		frames,
		x0,
		z0,
		block_w,
		block_h);
	CUDA_CHECK(cudaGetLastError());
}

void ScatterIqBlockFrameMajor(
	const float2* d_block,
	float2* d_full,
	int img_nx,
	int img_nz,
	int frames,
	int x0,
	int z0,
	int block_w,
	int block_h,
	cudaStream_t stream)
{
	ValidateBlockArgs(d_block, d_full, img_nx, img_nz, frames, x0, z0, block_w, block_h);
	const int total = block_w * block_h * frames;
	const int threads = 256;
	const int blocks = (total + threads - 1) / threads;
	ScatterIqBlockFrameMajorKernel<<<blocks, threads, 0, stream>>>(
		d_block,
		d_full,
		img_nx,
		img_nz,
		frames,
		x0,
		z0,
		block_w,
		block_h);
	CUDA_CHECK(cudaGetLastError());
}
