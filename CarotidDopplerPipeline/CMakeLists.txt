cmake_minimum_required(VERSION 3.27)

project(CarotidDopplerPipeline VERSION 0.1.0 LANGUAGES CXX CUDA)

find_package(Git QUIET)
set(CAROTID_GIT_COMMIT "unavailable")
if(Git_FOUND)
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" rev-parse --short=12 HEAD
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        RESULT_VARIABLE CAROTID_GIT_RESULT
        OUTPUT_VARIABLE CAROTID_GIT_COMMIT_CANDIDATE
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(CAROTID_GIT_RESULT EQUAL 0 AND NOT CAROTID_GIT_COMMIT_CANDIDATE STREQUAL "")
        set(CAROTID_GIT_COMMIT "${CAROTID_GIT_COMMIT_CANDIDATE}")
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" status --porcelain -- .
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            OUTPUT_VARIABLE CAROTID_GIT_STATUS
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(NOT CAROTID_GIT_STATUS STREQUAL "")
            string(APPEND CAROTID_GIT_COMMIT "-dirty")
        endif()
    endif()
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

find_package(CUDAToolkit REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(CAROTID_CUDA_ARCHITECTURES "89" CACHE STRING
    "CUDA architectures used to build the numeric generator")

add_executable(carotid_doppler_cuda
    src/config.cpp
    src/main.cpp
    src/pipeline_runner.cpp
    src/rf_reader.cpp
    src/run_manifest.cpp
    src/synthetic_generator.cpp
    src/cuda/synthetic_generator.cu
)

target_include_directories(carotid_doppler_cuda PRIVATE include)
target_link_libraries(carotid_doppler_cuda PRIVATE CUDA::cudart)
target_compile_definitions(carotid_doppler_cuda PRIVATE
    CAROTID_BUILD_TYPE="$<CONFIG>"
    CAROTID_GIT_COMMIT="${CAROTID_GIT_COMMIT}"
    CAROTID_PROJECT_VERSION="${PROJECT_VERSION}"
)

target_compile_options(carotid_doppler_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive->
    $<$<COMPILE_LANGUAGE:CUDA>:-allow-unsupported-compiler -Xcompiler=/W4,/wd4505>
)

set_target_properties(carotid_doppler_cuda PROPERTIES
    CUDA_ARCHITECTURES "${CAROTID_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION ON
)

add_library(carotid_global_svd STATIC
    src/cuda/global_temporal_svd.cu
)

target_include_directories(carotid_global_svd PUBLIC include)
target_link_libraries(carotid_global_svd PUBLIC
    CUDA::cudart
    CUDA::cublas
    CUDA::cusolver
)
target_compile_options(carotid_global_svd PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-allow-unsupported-compiler -Xcompiler=/W4,/wd4505>
)
set_target_properties(carotid_global_svd PROPERTIES
    CUDA_ARCHITECTURES "${CAROTID_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

add_executable(carotid_global_svd_diagnostic
    src/global_svd_diagnostic_main.cpp
)
target_include_directories(carotid_global_svd_diagnostic PRIVATE include)
target_link_libraries(carotid_global_svd_diagnostic PRIVATE carotid_global_svd)
target_compile_options(carotid_global_svd_diagnostic PRIVATE /W4 /permissive-)

add_library(carotid_rf_iq STATIC
    src/rf_bandpass.cpp
    src/cuda/rf_bandpass.cu
    src/cuda/rf_to_iq.cu
)
target_include_directories(carotid_rf_iq PUBLIC include)
target_link_libraries(carotid_rf_iq PUBLIC CUDA::cudart CUDA::cufft)
target_compile_options(carotid_rf_iq PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive->
    $<$<COMPILE_LANGUAGE:CUDA>:-allow-unsupported-compiler -Xcompiler=/W4,/wd4505>
)
set_target_properties(carotid_rf_iq PROPERTIES
    CUDA_ARCHITECTURES "${CAROTID_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

add_executable(carotid_iq_diagnostic
    src/config.cpp
    src/iq_diagnostic_main.cpp
    src/rf_reader.cpp
)
target_include_directories(carotid_iq_diagnostic PRIVATE include)
target_link_libraries(carotid_iq_diagnostic PRIVATE carotid_rf_iq)
target_compile_options(carotid_iq_diagnostic PRIVATE /W4 /permissive-)

add_library(carotid_das STATIC
    src/cuda/das_beamformer.cu
)
target_include_directories(carotid_das PUBLIC include)
target_link_libraries(carotid_das PUBLIC CUDA::cudart)
target_compile_options(carotid_das PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-allow-unsupported-compiler -Xcompiler=/W4,/wd4505>
)
set_target_properties(carotid_das PROPERTIES
    CUDA_ARCHITECTURES "${CAROTID_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_link_libraries(carotid_doppler_cuda PRIVATE
    carotid_rf_iq
    carotid_das
    carotid_global_svd
)

add_executable(carotid_das_diagnostic
    src/config.cpp
    src/das_diagnostic_main.cpp
    src/rf_reader.cpp
)
target_include_directories(carotid_das_diagnostic PRIVATE include)
target_link_libraries(carotid_das_diagnostic PRIVATE
    carotid_rf_iq
    carotid_das
)
target_compile_options(carotid_das_diagnostic PRIVATE /W4 /permissive-)

include(CTest)
if(BUILD_TESTING)
    add_test(NAME carotid_doppler_cuda_help COMMAND carotid_doppler_cuda --help)
    add_test(
        NAME carotid_global_svd_diagnostic_help
        COMMAND carotid_global_svd_diagnostic --help
    )
    add_test(
        NAME carotid_doppler_python_contracts
        COMMAND ${Python3_EXECUTABLE} -m unittest discover
            -s tests/python -t . -v
    )
    set_tests_properties(carotid_doppler_python_contracts PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        ENVIRONMENT
            "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python;CAROTID_DOPPLER_CUDA_EXE=$<TARGET_FILE:carotid_doppler_cuda>;CAROTID_GLOBAL_SVD_DIAGNOSTIC_EXE=$<TARGET_FILE:carotid_global_svd_diagnostic>"
    )
endif()
