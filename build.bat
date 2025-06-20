@echo off
echo 软光栅渲染器编译脚本
echo ======================

if not exist build mkdir build
cd build

echo 配置项目...
cmake .. -G "Visual Studio 16 2019" -A x64
if errorlevel 1 (
    echo CMake配置失败！
    pause
    exit /b 1
)

echo 编译项目...
cmake --build . --config Release
if errorlevel 1 (
    echo 编译失败！
    pause
    exit /b 1
)

echo 编译完成！
echo 可执行文件位置: build\Release\SoftRenderer.exe
echo.
echo 运行程序...
cd Release
SoftRenderer.exe
pause 