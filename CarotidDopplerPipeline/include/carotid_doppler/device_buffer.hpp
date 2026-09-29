#pragma once

#include "carotid_doppler/cuda_error.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace carotid_doppler {

template <typename T>
class DeviceBuffer final {
public:
    DeviceBuffer() noexcept = default;

    DeviceBuffer(std::size_t element_count, std::string_view stage) {
        allocate(element_count, stage);
    }

    ~DeviceBuffer() noexcept {
        release_noexcept();
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)),
          byte_count_(std::exchange(other.byte_count_, 0)) {}

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            release_noexcept();
            data_ = std::exchange(other.data_, nullptr);
            byte_count_ = std::exchange(other.byte_count_, 0);
        }
        return *this;
    }

    [[nodiscard]] T* get() noexcept {
        return data_;
    }

    [[nodiscard]] const T* get() const noexcept {
        return data_;
    }

    [[nodiscard]] std::size_t bytes() const noexcept {
        return byte_count_;
    }

    void release(std::string_view stage) {
        if (data_ == nullptr) {
            return;
        }

        T* const allocation = std::exchange(data_, nullptr);
        const std::size_t allocation_bytes = std::exchange(byte_count_, 0);
        check_cuda(
            cudaFree(allocation),
            context(stage, "cudaFree", allocation_bytes));
    }

private:
    static std::string context(
        std::string_view stage,
        std::string_view operation,
        std::size_t bytes) {
        return std::string(stage) + " " + std::string(operation) +
            " (bytes=" + std::to_string(bytes) + ")";
    }

    void allocate(std::size_t element_count, std::string_view stage) {
        if (element_count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::overflow_error(
                std::string(stage) + " allocation byte count overflows size_t");
        }

        const std::size_t allocation_bytes = element_count * sizeof(T);
        T* allocation = nullptr;
        check_cuda(
            cudaMalloc(reinterpret_cast<void**>(&allocation), allocation_bytes),
            context(stage, "cudaMalloc", allocation_bytes));
        data_ = allocation;
        byte_count_ = allocation_bytes;
    }

    void release_noexcept() noexcept {
        if (data_ != nullptr) {
            (void)cudaFree(data_);
            data_ = nullptr;
            byte_count_ = 0;
        }
    }

    T* data_ = nullptr;
    std::size_t byte_count_ = 0;
};

}  // namespace carotid_doppler
