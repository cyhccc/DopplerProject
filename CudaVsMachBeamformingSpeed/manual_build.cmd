@echo off
setlocal
call "E:\Compiler\Visual  Studio\IDE\VC\Auxiliary\Build\vcvars64.bat"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
set "PROJ=%~dp0"
if "%PROJ:~-1%"=="\" set "PROJ=%PROJ:~0,-1%"
set "OUT=%PROJ%\manual_build"
if not exist "%OUT%" mkdir "%OUT%"
pushd "%PROJ%"
"%CUDA_PATH%\bin\nvcc.exe" -std=c++17 -O3 -arch=sm_89 -cudart shared -I"%CUDA_PATH%\include" -I"%PROJ%" -Xcompiler "/EHsc /O2 /MD" -c beamformer.cu -o "%OUT%\beamformer.obj"
if errorlevel 1 exit /b 1
cl /std:c++17 /EHsc /O2 /MD /I"%CUDA_PATH%\include" /I"%PROJ%" /c main.cpp /Fo"%OUT%\main.obj"
if errorlevel 1 exit /b 1
cl /std:c++17 /EHsc /O2 /MD /I"%CUDA_PATH%\include" /I"%PROJ%" /c params.cpp /Fo"%OUT%\params.obj"
if errorlevel 1 exit /b 1
link /OUT:"%OUT%\OurCudaBeamformBench.exe" "%OUT%\main.obj" "%OUT%\params.obj" "%OUT%\beamformer.obj" /LIBPATH:"%CUDA_PATH%\lib\x64" cudart.lib
if errorlevel 1 exit /b 1
popd
