#ifndef CONFIG_CUH
#define CONFIG_CUH
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<iostream>
#include<string>
#include <sstream> // 引入 stringstream 以便更灵活地构建字符串
#include<vector>
#include <cmath>
#include <cstdio>
#include <cstdarg> // 用于处理可变参数 va_list
#include <ctime>
#include <chrono>
#include <stdexcept>
#include <cufft.h>

inline std::string GetCurrentTimeStr() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    char buf[30];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now_c));
    return std::string(buf);
}

// 双重日志函数：同时输出到控制台和文件
// 格式化用法和 printf 一模一样：DualLog("Value: %d\n", val);
inline void DualLog(const char* format, ...) {
    // 1. 准备时间戳
    std::string time_str = GetCurrentTimeStr();

    // 2. 解析变长参数，格式化为字符串
    // 我们使用一个较大的 buffer 来存放格式化后的内容
    const int BUFFER_SIZE = 4096;
    char buffer[BUFFER_SIZE];

    va_list args;
    va_start(args, format);
    vsnprintf(buffer, BUFFER_SIZE, format, args); // 格式化到 buffer 中
    va_end(args);

    // 3. 输出到控制台
    // 这里的 buffer 已经包含了用户传进来的 \n，所以不需要额外加
    printf("[%s] %s", time_str.c_str(), buffer);

    // 4. 输出到文件
    FILE* fp = fopen("error_debug.log", "a");
    if (fp) {
        fprintf(fp, "[%s] %s", time_str.c_str(), buffer);
        fclose(fp);
    }
}

inline cudaError_t ErrorCheck(cudaError_t error_code, const char* filename, int lineNumber)
{
    if (error_code != cudaSuccess)
    {
        // 使用 stringstream 构建详细的错误信息
        std::stringstream ss;
        ss << "\n"
            << ">> FATAL CUDA ERROR DETECTED <<\n"
            << "--------------------------------------------------\n"
            << "  [Error Code] : " << error_code << "\n"
            << "  [Error Name] : " << cudaGetErrorName(error_code) << "\n" // 例如: cudaErrorMemoryAllocation
            << "  [Description]: " << cudaGetErrorString(error_code) << "\n" // 例如: out of memory
            << "  [File Location]: " << filename << "\n"
            << "  [Line Number] : " << lineNumber << "\n"
            << "--------------------------------------------------\n";

        // 将 stringstream 转为 string
        std::string msg = ss.str();
        DualLog("%s", msg.c_str());
        // 抛出异常，包含完整的 msg
        throw std::runtime_error(msg);
    }
    return error_code;
}
// 定义宏
#define CUDA_CHECK(call) ErrorCheck((call), __FILE__, __LINE__)


#ifdef _DEBUG
    // Debug 模式：强同步，查所有错误（慢，但能精准定位 bug）
#define CHECK_KERNEL() do { \
        CUDA_CHECK(cudaGetLastError()); \
        CUDA_CHECK(cudaDeviceSynchronize()); \
    } while(0)
#else
    // Release 模式：只查启动错误，不等待（快）
    // 或者干脆定义为空： #define CHECK_KERNEL() ((void)0)
//#define CHECK_KERNEL() CUDA_CHECK(cudaPeekAtLastError())
#define CHECK_KERNEL() ((void)0)
#endif




inline void TestValue(float* d_buffer, int nx, int nz)
{
    int total_pixels = nx * nz;
    size_t bytes = (size_t)total_pixels * sizeof(float);

    // 使用 vector 管理内存 (RAII)
    std::vector<float> h_buf(total_pixels);
    CUDA_CHECK(cudaMemcpy(h_buf.data(), d_buffer, bytes, cudaMemcpyDeviceToHost));

    // 统计分析
    float min_val = 1e30f;
    float max_val = -1e30f;
    double sum_val = 0.0;
    int zero_count = 0;
    int nan_count = 0;

    for (int i = 0; i < total_pixels; i++) {
        float val = h_buf[i];
        if (std::isnan(val)) { nan_count++; continue; }
        if (val == 0.0f) zero_count++;
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
        sum_val += (double)val;
    }
    std::string msg;
    if (nan_count > 0) {
        msg = "Diagnosis: [ERROR] Data contains NaN!\n";
        DualLog(msg.c_str());
        throw std::runtime_error(msg);
    }
    else if (max_val == 0.0f) {
        msg = "Diagnosis: [ERROR] All Zeros. No signal.\n";
        DualLog(msg.c_str());
        throw std::runtime_error(msg);
    }
    else if (max_val < 1e-9f) {
        msg = "Diagnosis: [ERROR] Signal too weak.\n";
        DualLog(msg.c_str());
        throw std::runtime_error(msg);
    }
    else {
        msg = "Diagnosis: [OK] Signal detected.\n";
        std::cout << msg;
    }
}


inline const char* GetCuFFTErrorString(cufftResult error)
{
    switch (error)
    {
    case CUFFT_SUCCESS: return "CUFFT_SUCCESS";
    case CUFFT_INVALID_PLAN: return "CUFFT_INVALID_PLAN";
    case CUFFT_ALLOC_FAILED: return "CUFFT_ALLOC_FAILED";
    case CUFFT_INVALID_TYPE: return "CUFFT_INVALID_TYPE";
    case CUFFT_INVALID_VALUE: return "CUFFT_INVALID_VALUE";
    case CUFFT_INTERNAL_ERROR: return "CUFFT_INTERNAL_ERROR";
    case CUFFT_EXEC_FAILED: return "CUFFT_EXEC_FAILED";
    case CUFFT_SETUP_FAILED: return "CUFFT_SETUP_FAILED";
    case CUFFT_INVALID_SIZE: return "CUFFT_INVALID_SIZE";
    case CUFFT_UNALIGNED_DATA: return "CUFFT_UNALIGNED_DATA";
    case CUFFT_INCOMPLETE_PARAMETER_LIST: return "CUFFT_INCOMPLETE_PARAMETER_LIST";
    case CUFFT_INVALID_DEVICE: return "CUFFT_INVALID_DEVICE";
    case CUFFT_PARSE_ERROR: return "CUFFT_PARSE_ERROR";
    case CUFFT_NO_WORKSPACE: return "CUFFT_NO_WORKSPACE";
    case CUFFT_NOT_IMPLEMENTED: return "CUFFT_NOT_IMPLEMENTED";
    case CUFFT_LICENSE_ERROR: return "CUFFT_LICENSE_ERROR";
    case CUFFT_NOT_SUPPORTED: return "CUFFT_NOT_SUPPORTED";
    default: return "Unknown cuFFT Error";
    }
}

// ==========================================
// [新增] cuFFT 专用的检查函数
// ==========================================
inline cufftResult CuFFTCheck(cufftResult err, const char* file, int line)
{
    if (err != CUFFT_SUCCESS)
    {
        std::stringstream ss;
        ss << "\n>> FATAL CUFFT ERROR <<\n"
            << "Code: " << err << " (" << GetCuFFTErrorString(err) << ")\n"
            << "File: " << file << ":" << line << "\n";

        std::string msg = ss.str();

        // 使用你的 DualLog 记录到文件和控制台
        DualLog("%s", msg.c_str());

        // 抛出运行时异常
        throw std::runtime_error(msg);
    }
    return err;
}

// ==========================================
// [新增] 定义宏 CUFFT_CHECK
// ==========================================
#define CUFFT_CHECK(call) CuFFTCheck((call), __FILE__, __LINE__)


#endif // CONFIG_CUH



