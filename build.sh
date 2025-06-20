#!/bin/bash

echo "软光栅渲染器编译脚本"
echo "======================"

# 创建构建目录
if [ ! -d "build" ]; then
    mkdir build
fi
cd build

# 配置项目
echo "配置项目..."
cmake ..
if [ $? -ne 0 ]; then
    echo "CMake配置失败！"
    exit 1
fi

# 编译项目
echo "编译项目..."
make -j$(nproc)
if [ $? -ne 0 ]; then
    echo "编译失败！"
    exit 1
fi

echo "编译完成！"
echo "可执行文件位置: build/SoftRenderer"
echo ""
echo "运行程序..."
./SoftRenderer 