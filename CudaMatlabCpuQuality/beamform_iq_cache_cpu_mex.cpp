#include "mex.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

namespace {

class MappedFile {
public:
    explicit MappedFile(const std::wstring& path) {
        file_ = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file_ == INVALID_HANDLE_VALUE) {
            mexErrMsgIdAndTxt("beamform_cpu_mex:OpenFailed", "Cannot open IQ file (Win32 error %lu).", GetLastError());
        }

        LARGE_INTEGER size{};
        if (!GetFileSizeEx(file_, &size)) {
            const DWORD error = GetLastError();
            CloseHandle(file_);
            file_ = INVALID_HANDLE_VALUE;
            mexErrMsgIdAndTxt("beamform_cpu_mex:SizeFailed", "Cannot query IQ file size (Win32 error %lu).", error);
        }
        size_ = static_cast<std::uint64_t>(size.QuadPart);

        mapping_ = CreateFileMappingW(file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (mapping_ == nullptr) {
            const DWORD error = GetLastError();
            CloseHandle(file_);
            file_ = INVALID_HANDLE_VALUE;
            mexErrMsgIdAndTxt("beamform_cpu_mex:MapFailed", "Cannot create IQ file mapping (Win32 error %lu).", error);
        }
        data_ = MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, 0);
        if (data_ == nullptr) {
            const DWORD error = GetLastError();
            CloseHandle(mapping_);
            CloseHandle(file_);
            mapping_ = nullptr;
            file_ = INVALID_HANDLE_VALUE;
            mexErrMsgIdAndTxt("beamform_cpu_mex:ViewFailed", "Cannot map IQ file (Win32 error %lu).", error);
        }
    }

    ~MappedFile() {
        if (data_ != nullptr) UnmapViewOfFile(data_);
        if (mapping_ != nullptr) CloseHandle(mapping_);
        if (file_ != INVALID_HANDLE_VALUE) CloseHandle(file_);
    }

    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    const float* floats() const { return static_cast<const float*>(data_); }
    std::uint64_t size() const { return size_; }

private:
    HANDLE file_ = INVALID_HANDLE_VALUE;
    HANDLE mapping_ = nullptr;
    void* data_ = nullptr;
    std::uint64_t size_ = 0;
};

void require(bool condition, const char* id, const char* message) {
    if (!condition) mexErrMsgIdAndTxt(id, "%s", message);
}

std::wstring matlabStringToWide(const mxArray* value) {
    require(mxIsChar(value), "beamform_cpu_mex:PathType", "bin_path must be a character vector.");
    const mwSize length = mxGetNumberOfElements(value);
    std::vector<wchar_t> buffer(length + 1);
    const mxChar* chars = mxGetChars(value);
    for (mwSize i = 0; i < length; ++i) buffer[i] = static_cast<wchar_t>(chars[i]);
    buffer[length] = L'\0';
    return std::wstring(buffer.data(), length);
}

mwSize positiveIntegerScalar(const mxArray* value, const char* name) {
    require(mxIsNumeric(value) && !mxIsComplex(value) && mxGetNumberOfElements(value) == 1,
            "beamform_cpu_mex:ScalarType", "Frame and sample counts must be real numeric scalars.");
    const double number = mxGetScalar(value);
    if (!(number >= 1.0 && number <= static_cast<double>(std::numeric_limits<mwSize>::max()) && std::floor(number) == number)) {
        mexErrMsgIdAndTxt("beamform_cpu_mex:ScalarValue", "%s must be a positive integer.", name);
    }
    return static_cast<mwSize>(number);
}

} // namespace

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    require(nrhs == 7 || nrhs == 9, "beamform_cpu_mex:ArgumentCount",
            "Usage: iq = beamform_iq_cache_cpu_mex(bin_path, rx_delay, rx_weight, tx_delay, phase_scale, n_frames, n_samples [, phase_cos, phase_sin])");
    require(nlhs <= 1, "beamform_cpu_mex:OutputCount", "One output is supported.");

    const mxArray* rx_delay_array = prhs[1];
    const mxArray* rx_weight_array = prhs[2];
    const mxArray* tx_delay_array = prhs[3];
    require(mxIsSingle(rx_delay_array) && !mxIsComplex(rx_delay_array) && mxGetNumberOfDimensions(rx_delay_array) == 2,
            "beamform_cpu_mex:RxDelayType", "rx_delay must be a real single matrix [pixels, channels].");
    require(mxIsLogical(rx_weight_array) && mxGetNumberOfDimensions(rx_weight_array) == 2,
            "beamform_cpu_mex:RxWeightType", "rx_weight must be a logical matrix [pixels, channels].");
    require(mxIsSingle(tx_delay_array) && !mxIsComplex(tx_delay_array) && mxGetNumberOfDimensions(tx_delay_array) == 2,
            "beamform_cpu_mex:TxDelayType", "tx_delay must be a real single matrix [pixels, angles].");
    require(mxIsSingle(prhs[4]) && !mxIsComplex(prhs[4]) && mxGetNumberOfElements(prhs[4]) == 1,
            "beamform_cpu_mex:PhaseScaleType", "phase_scale must be a real single scalar.");

    const mwSize n_pixels = mxGetM(rx_delay_array);
    const mwSize n_channels = mxGetN(rx_delay_array);
    const mwSize n_angles = mxGetN(tx_delay_array);
    require(mxGetM(tx_delay_array) == n_pixels, "beamform_cpu_mex:TxDelaySize", "tx_delay pixel count differs from rx_delay.");
    require(mxGetM(rx_weight_array) == n_pixels && mxGetN(rx_weight_array) == n_channels,
            "beamform_cpu_mex:RxWeightSize", "rx_weight size must match rx_delay.");

    const mwSize n_frames = positiveIntegerScalar(prhs[5], "n_frames");
    const mwSize n_samples = positiveIntegerScalar(prhs[6], "n_samples");
    const float* phase_cos = nullptr;
    const float* phase_sin = nullptr;
    if (nrhs == 9) {
        const mwSize phase_elements = n_pixels * n_channels * n_angles;
        require(mxIsSingle(prhs[7]) && !mxIsComplex(prhs[7]) && mxGetNumberOfElements(prhs[7]) == phase_elements,
                "beamform_cpu_mex:PhaseCosSize", "phase_cos must be real single with pixels*channels*angles elements.");
        require(mxIsSingle(prhs[8]) && !mxIsComplex(prhs[8]) && mxGetNumberOfElements(prhs[8]) == phase_elements,
                "beamform_cpu_mex:PhaseSinSize", "phase_sin must be real single with pixels*channels*angles elements.");
        phase_cos = mxGetSingles(prhs[7]);
        phase_sin = mxGetSingles(prhs[8]);
    }
    const std::uint64_t expected_bytes = 2ULL * n_frames * n_samples * n_channels * n_angles * sizeof(float);

    MappedFile input(matlabStringToWide(prhs[0]));
    if (input.size() != expected_bytes) {
        mexErrMsgIdAndTxt("beamform_cpu_mex:FileSize",
                          "Unexpected IQ file size: expected %llu bytes, got %llu.",
                          static_cast<unsigned long long>(expected_bytes),
                          static_cast<unsigned long long>(input.size()));
    }

    const float* raw = input.floats();
    const float* rx_delay = mxGetSingles(rx_delay_array);
    const mxLogical* rx_weight = mxGetLogicals(rx_weight_array);
    const float* tx_delay = mxGetSingles(tx_delay_array);
    const float phase_scale = *mxGetSingles(prhs[4]);

    const mwSize output_dims[2] = {n_pixels, n_frames};
    plhs[0] = mxCreateNumericArray(2, output_dims, mxSINGLE_CLASS, mxCOMPLEX);
    mxComplexSingle* output = mxGetComplexSingles(plhs[0]);

    #pragma omp parallel
    {
        std::vector<float> acc_real(n_frames);
        std::vector<float> acc_imag(n_frames);

        #pragma omp for schedule(static)
        for (std::int64_t pixel_signed = 0; pixel_signed < static_cast<std::int64_t>(n_pixels); ++pixel_signed) {
            const mwSize pixel = static_cast<mwSize>(pixel_signed);
            std::fill(acc_real.begin(), acc_real.end(), 0.0f);
            std::fill(acc_imag.begin(), acc_imag.end(), 0.0f);

            // This nesting deliberately matches MATLAB: angle outer, channel inner.
            for (mwSize angle = 0; angle < n_angles; ++angle) {
                const float tx = tx_delay[pixel + angle * n_pixels];
                for (mwSize channel = 0; channel < n_channels; ++channel) {
                    const mwSize matrix_index = pixel + channel * n_pixels;
                    if (!rx_weight[matrix_index]) continue;

                    const float delay = tx + rx_delay[matrix_index];
                    if (!(delay >= 0.0f && delay < static_cast<float>(n_samples - 1))) continue;

                    const mwSize sample0 = static_cast<mwSize>(std::floor(delay));
                    const float fraction = delay - static_cast<float>(sample0);
                    const mwSize phase_index = matrix_index + angle * n_pixels * n_channels;
                    const float phase_arg = phase_scale * delay;
                    const float phase_real = phase_cos == nullptr ? std::cos(phase_arg) : phase_cos[phase_index];
                    const float phase_imag = phase_sin == nullptr ? std::sin(phase_arg) : phase_sin[phase_index];
                    const std::uint64_t trace_base = 2ULL * n_frames *
                        (sample0 + static_cast<std::uint64_t>(n_samples) *
                        (channel + static_cast<std::uint64_t>(n_channels) * angle));
                    const float* sample_a = raw + trace_base;
                    const float* sample_b = sample_a + 2ULL * n_frames;

                    for (mwSize frame = 0; frame < n_frames; ++frame) {
                        const float a_real = sample_a[2 * frame];
                        const float a_imag = sample_a[2 * frame + 1];
                        const float delayed_real = a_real + (sample_b[2 * frame] - a_real) * fraction;
                        const float delayed_imag = a_imag + (sample_b[2 * frame + 1] - a_imag) * fraction;
                        const float rotated_real = delayed_real * phase_real - delayed_imag * phase_imag;
                        const float rotated_imag = delayed_real * phase_imag + delayed_imag * phase_real;
                        acc_real[frame] += rotated_real;
                        acc_imag[frame] += rotated_imag;
                    }
                }
            }

            for (mwSize frame = 0; frame < n_frames; ++frame) {
                mxComplexSingle& value = output[pixel + frame * n_pixels];
                value.real = acc_real[frame];
                value.imag = acc_imag[frame];
            }
        }
    }
}
