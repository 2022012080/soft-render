cmake_minimum_required(VERSION 3.10)
project(SoftRenderer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置编译选项
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")
endif()

# 包含目录
include_directories(${CMAKE_SOURCE_DIR}/include)

# 源文件
set(SOURCES
    src/main.cpp
    src/renderer.cpp
    src/model.cpp
    src/texture.cpp
    src/math.cpp
)

# 创建可执行文件
add_executable(SoftRenderer ${SOURCES})

# 复制资源文件
file(COPY assets DESTINATION ${CMAKE_BINARY_DIR}) 