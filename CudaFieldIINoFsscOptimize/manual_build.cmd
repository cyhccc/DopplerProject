@echo off
setlocal
call "E:\Compiler\Visual  Studio\IDE\VC\Auxiliary\Build\vcvars64.bat"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
set "PROJ=%~dp0"
if "%PROJ:~-1%"=="\" set "PROJ=%PROJ:~0,-1%"
set "OUT=%PROJ%\manual_build"
if not exist "%OUT%" mkdir "%OUT%"
pushd "%PROJ%"
"%CUDA_PATH%\bin\nvcc.exe" -std=c++17 -O3 --use_fast_math -lineinfo -Xptxas -v -arch=sm_89 -cudart shared -I"%CUDA_PATH%\include" -I"%PROJ%" -Xcompiler "/EHsc /O2 /MD" -c beamformer.cu -o "%OUT%\beamformer.obj"
if errorlevel 1 exit /b 1
"%CUDA_PATH%\bin\nvcc.exe" -std=c++17 -O3 --use_fast_math -lineinfo -Xptxas -v -arch=sm_89 -cudart shared -I"%CUDA_PATH%\include" -I"%PROJ%" -Xcompiler "/EHsc /O2 /MD" -c data_preprocess_gpu_v2.cu -o "%OUT%\data_preprocess_gpu_v2.obj"
if errorlevel 1 exit /b 1
"%CUDA_PATH%\bin\nvcc.exe" -std=c++17 -O3 --use_fast_math -lineinfo -Xptxas -v -arch=sm_89 -cudart shared -I"%CUDA_PATH%\include" -I"%PROJ%" -I"%PROJ%\..\CudaMatlabCpuQuality" -Xcompiler "/EHsc /O2 /MD" -c "%PROJ%\..\CudaMatlabCpuQuality\rsvd_filter.cu" -o "%OUT%\rsvd_filter.obj"
if errorlevel 1 exit /b 1
"%CUDA_PATH%\bin\nvcc.exe" -std=c++17 -O3 --use_fast_math -lineinfo -Xptxas -v -arch=sm_89 -cudart shared -I"%CUDA_PATH%\include" -I"%PROJ%" -Xcompiler "/EHsc /O2 /MD" -c rf_bandpass_filter.cu -o "%OUT%\rf_bandpass_filter.obj"
if errorlevel 1 exit /b 1
"%CUDA_PATH%\bin\nvcc.exe" -std=c++17 -O3 --use_fast_math -lineinfo -Xptxas -v -arch=sm_89 -cudart shared -I"%CUDA_PATH%\include" -I"%PROJ%" -Xcompiler "/EHsc /O2 /MD" -c blockwise_svd.cu -o "%OUT%\blockwise_svd.obj"
if errorlevel 1 exit /b 1
cl /std:c++17 /EHsc /O2 /MD /I"%CUDA_PATH%\include" /I"%PROJ%" /c main.cpp /Fo"%OUT%\main.obj"
if errorlevel 1 exit /b 1
cl /std:c++17 /EHsc /O2 /MD /I"%CUDA_PATH%\include" /I"%PROJ%" /c params.cpp /Fo"%OUT%\params.obj"
if errorlevel 1 exit /b 1
cl /std:c++17 /EHsc /O2 /MD /I"%CUDA_PATH%\include" /I"%PROJ%" /c motion_correction.cpp /Fo"%OUT%\motion_correction.obj"
if errorlevel 1 exit /b 1
cl /std:c++17 /EHsc /O2 /MD /I"%CUDA_PATH%\include" /I"%PROJ%" /c test_blockwise_svd.cpp /Fo"%OUT%\test_blockwise_svd.obj"
if errorlevel 1 exit /b 1
link /OUT:"%OUT%\test_blockwise_svd.exe" "%OUT%\test_blockwise_svd.obj" "%OUT%\blockwise_svd.obj" /LIBPATH:"%CUDA_PATH%\lib\x64" cudart.lib
if errorlevel 1 exit /b 1
link /OUT:"%OUT%\FieldIIPwCudaBeamformBench.exe" "%OUT%\main.obj" "%OUT%\params.obj" "%OUT%\beamformer.obj" "%OUT%\data_preprocess_gpu_v2.obj" /LIBPATH:"%CUDA_PATH%\lib\x64" cudart.lib cufft.lib
if errorlevel 1 exit /b 1
cl /std:c++17 /EHsc /O2 /MD /I"%CUDA_PATH%\include" /I"%PROJ%" /I"%PROJ%\..\CudaMatlabCpuQuality" /c main_jindongmai_doppler.cpp /Fo"%OUT%\main_jindongmai_doppler.obj"
if errorlevel 1 exit /b 1
link /OUT:"%OUT%\JindongmaiDoppler.exe" "%OUT%\main_jindongmai_doppler.obj" "%OUT%\motion_correction.obj" "%OUT%\params.obj" "%OUT%\beamformer.obj" "%OUT%\data_preprocess_gpu_v2.obj" "%OUT%\rsvd_filter.obj" "%OUT%\rf_bandpass_filter.obj" "%OUT%\blockwise_svd.obj" /LIBPATH:"%CUDA_PATH%\lib\x64" cudart.lib cufft.lib cublas.lib cusolver.lib
if errorlevel 1 exit /b 1
popd
