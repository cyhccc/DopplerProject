#ifndef BLOCKWISE_SVD_CUH
#define BLOCKWISE_SVD_CUH

#include <cuda_runtime.h>

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
	cudaStream_t stream = 0);

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
	cudaStream_t stream = 0);

#endif
