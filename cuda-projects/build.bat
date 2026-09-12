@echo off
REM Build a project:  build.bat 05-monte-carlo\monte_carlo.cu
REM Output lands in .\bin\
setlocal

if "%~1"=="" (
    echo Usage: build.bat ^<path-to-.cu^> [extra nvcc flags]
    exit /b 1
)
if not exist "%~1" (
    echo ERROR: no such file: %~1
    exit /b 1
)

set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" (
    echo ERROR: MSVC not found at "%VCVARS%"
    exit /b 1
)
call "%VCVARS%" >nul 2>&1

if not defined CUDA_PATH (
    for /d %%d in ("C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*") do set "CUDA_PATH=%%d"
)
if exist "%CUDA_PATH%\bin\nvcc.exe" set "PATH=%CUDA_PATH%\bin;%PATH%"

where nvcc >nul 2>&1
if errorlevel 1 (
    echo ERROR: nvcc not found. Is the CUDA Toolkit installed?
    exit /b 1
)

if not exist bin mkdir bin

REM CUDA 13 moved thrust/cub under include\cccl, so add it explicitly.
set "EXTRA=%~2 %~3 %~4 %~5"

nvcc -arch=sm_75 -O3 -std=c++17 -lineinfo --extended-lambda -Xcompiler /Zc:preprocessor ^
     -I"%CUDA_PATH%\include\cccl" ^
     -lcurand -lcublas ^
     "%~1" -o "bin\%~n1.exe" %EXTRA%
if errorlevel 1 (
    echo BUILD FAILED
    exit /b 1
)
echo Built bin\%~n1.exe
