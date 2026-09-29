#include "blockwise_svd.cuh"
#include "config.cuh"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
	void ExpectFloat2Near(const float2& actual, const float2& expected, const std::string& label)
	{
		const float eps = 1.0e-6f;
		if (std::fabs(actual.x - expected.x) > eps || std::fabs(actual.y - expected.y) > eps)
		{
			throw std::runtime_error(label + " mismatch: got (" + std::to_string(actual.x) + ","
				+ std::to_string(actual.y) + "), expected (" + std::to_string(expected.x) + ","
				+ std::to_string(expected.y) + ")");
		}
	}
}

int main()
{
	const int img_nx = 5;
	const int img_nz = 4;
	const int frames = 3;
	const int x0 = 1;
	const int z0 = 1;
	const int block_w = 3;
	const int block_h = 2;
	const int pixels = img_nx * img_nz;
	const int block_pixels = block_w * block_h;

	std::vector<float2> full(static_cast<size_t>(frames) * pixels);
	for (int frame = 0; frame < frames; ++frame)
	{
		for (int pixel = 0; pixel < pixels; ++pixel)
		{
			const float value = static_cast<float>(frame * 1000 + pixel);
			full[static_cast<size_t>(frame) * pixels + pixel] = make_float2(value, -value);
		}
	}

	float2* d_full = nullptr;
	float2* d_block = nullptr;
	float2* d_roundtrip = nullptr;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_full), full.size() * sizeof(float2)));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_block), static_cast<size_t>(frames) * block_pixels * sizeof(float2)));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_roundtrip), full.size() * sizeof(float2)));

	try
	{
		CUDA_CHECK(cudaMemcpy(d_full, full.data(), full.size() * sizeof(float2), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemset(d_roundtrip, 0, full.size() * sizeof(float2)));

		GatherIqBlockFrameMajor(d_full, d_block, img_nx, img_nz, frames, x0, z0, block_w, block_h);
		CUDA_CHECK(cudaDeviceSynchronize());

		std::vector<float2> block(static_cast<size_t>(frames) * block_pixels);
		CUDA_CHECK(cudaMemcpy(block.data(), d_block, block.size() * sizeof(float2), cudaMemcpyDeviceToHost));
		for (int frame = 0; frame < frames; ++frame)
		{
			for (int lz = 0; lz < block_h; ++lz)
			{
				for (int lx = 0; lx < block_w; ++lx)
				{
					const int local = lz * block_w + lx;
					const int global_pixel = (z0 + lz) * img_nx + (x0 + lx);
					ExpectFloat2Near(
						block[static_cast<size_t>(frame) * block_pixels + local],
						full[static_cast<size_t>(frame) * pixels + global_pixel],
						"gather frame=" + std::to_string(frame) + " local=" + std::to_string(local));
				}
			}
		}

		ScatterIqBlockFrameMajor(d_block, d_roundtrip, img_nx, img_nz, frames, x0, z0, block_w, block_h);
		CUDA_CHECK(cudaDeviceSynchronize());

		std::vector<float2> roundtrip(full.size());
		CUDA_CHECK(cudaMemcpy(roundtrip.data(), d_roundtrip, roundtrip.size() * sizeof(float2), cudaMemcpyDeviceToHost));
		for (int frame = 0; frame < frames; ++frame)
		{
			for (int z = 0; z < img_nz; ++z)
			{
				for (int x = 0; x < img_nx; ++x)
				{
					const int pixel = z * img_nx + x;
					const bool inside = x >= x0 && x < x0 + block_w && z >= z0 && z < z0 + block_h;
					const float2 expected = inside
						? full[static_cast<size_t>(frame) * pixels + pixel]
						: make_float2(0.0f, 0.0f);
					ExpectFloat2Near(
						roundtrip[static_cast<size_t>(frame) * pixels + pixel],
						expected,
						"scatter frame=" + std::to_string(frame) + " pixel=" + std::to_string(pixel));
				}
			}
		}

		CUDA_CHECK(cudaFree(d_roundtrip));
		d_roundtrip = nullptr;
		CUDA_CHECK(cudaFree(d_block));
		d_block = nullptr;
		CUDA_CHECK(cudaFree(d_full));
		d_full = nullptr;
		std::cout << "[Test] blockwise_svd gather/scatter passed\n";
	}
	catch (...)
	{
		if (d_roundtrip != nullptr) { cudaFree(d_roundtrip); }
		if (d_block != nullptr) { cudaFree(d_block); }
		if (d_full != nullptr) { cudaFree(d_full); }
		throw;
	}

	return 0;
}
