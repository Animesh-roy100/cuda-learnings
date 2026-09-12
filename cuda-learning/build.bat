@echo off
REM Build any example:   build.bat 03-vector-add\vector_add.cu
REM The exe lands in .\bin\
REM
REM sm_75 is Turing = your GTX 1650. Compiling for your exact architecture
REM skips JIT compilation at startup and enables all sm_75 features.

setlocal

if "%~1"=="" (
    echo Usage: build.bat ^<path-to-.cu^>
    echo   e.g. build.bat 00-check\driver_check.cu
    exit /b 1
)

if not exist "%~1" (
    echo ERROR: no such file: %~1
    exit /b 1
)

REM nvcc needs MSVC's cl.exe as its host compiler on Windows.
set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" (
    echo ERROR: MSVC not found at:
    echo   "%VCVARS%"
    echo Install "Desktop development with C++" via the Visual Studio Installer.
    exit /b 1
)
call "%VCVARS%" >nul 2>&1

REM The CUDA installer puts nvcc on the machine PATH, but a shell opened
REM BEFORE the install still has a stale environment. Find the toolkit
REM ourselves so this works either way.
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

nvcc -arch=sm_75 -O2 -lineinfo "%~1" -o "bin\%~n1.exe"
if errorlevel 1 (
    echo BUILD FAILED
    exit /b 1
)

echo Built bin\%~n1.exe
