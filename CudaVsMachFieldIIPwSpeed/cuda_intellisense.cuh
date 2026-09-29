#pragma once

#ifdef __INTELLISENSE__
#define CUDA_KERNEL_LAUNCH(kernel, grid, block, shared, stream, ...) kernel(__VA_ARGS__)
#else
#define CUDA_KERNEL_LAUNCH(kernel, grid, block, shared, stream, ...) kernel<<<grid, block, shared, stream>>>(__VA_ARGS__)
#endif
