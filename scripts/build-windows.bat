@echo off
REM Windows build script for Visual Studio
REM Run this from Visual Studio Developer Command Prompt

setlocal EnableDelayedExpansion

echo === C Transformer Windows Build Script (Visual Studio) ===

REM Check if we're in a Visual Studio environment
where cl >nul 2>&1
if errorlevel 1 (
    echo Error: cl.exe not found. Please run this from Visual Studio Developer Command Prompt.
    echo Or install Visual Studio Build Tools with C++ support.
    pause
    exit /b 1
)

echo Visual Studio C++ compiler found.

REM Check for vcpkg (recommended for dependencies)
if not defined VCPKG_ROOT (
    echo Warning: VCPKG_ROOT not set. Consider installing vcpkg for easier dependency management.
    echo Instructions:
    echo   1. git clone https://github.com/Microsoft/vcpkg.git
    echo   2. cd vcpkg && .\bootstrap-vcpkg.bat
    echo   3. set VCPKG_ROOT=^<path-to-vcpkg^>
    echo   4. vcpkg install pthreads:x64-windows
    echo.
)

echo Building with CMake...

REM Create build directory
if not exist build-vs mkdir build-vs
cd build-vs

REM Configure for Visual Studio
if defined VCPKG_ROOT (
    cmake .. -DCMAKE_TOOLCHAIN_FILE="%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake" -DCMAKE_BUILD_TYPE=Release
) else (
    cmake .. -DCMAKE_BUILD_TYPE=Release
)

if errorlevel 1 (
    echo Error during CMake configuration
    pause
    exit /b 1
)

REM Build
cmake --build . --config Release

if errorlevel 1 (
    echo Error during build
    pause
    exit /b 1
)

echo.
echo ✓ Build completed successfully!
echo Executable: %cd%\Release\Aaryan.exe
echo.
echo To test the build:
echo   Release\Aaryan --help

pause
