@echo off
echo Starting clean build process...

REM Clean existing build directory
if exist build (
    echo Cleaning previous build...
    rmdir /s /q build
)

REM Create new build directory
mkdir build
cd build

REM Configure with CMake
echo Configuring with CMake...
cmake .. -G "Visual Studio 17 2022" -A x64

REM Check if configuration was successful
if %ERRORLEVEL% NEQ 0 (
    echo CMake configuration failed!
    pause
    exit /b 1
)

REM Build the project
echo Building project...
cmake --build . --config Release

REM Check if build was successful
if %ERRORLEVEL% NEQ 0 (
    echo Build failed!
    pause
    exit /b 1
)

echo Build completed successfully!
echo Executable is in: build\Release\SoftRenderer.exe

cd ..
pause 