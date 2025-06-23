cmake_minimum_required(VERSION 3.10)
project(SoftRenderer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置编译选项
if(MSVC)
    # 使用UTF-8编码，只使用兼容的选项
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3 /utf-8")
    
    # 添加MSVC必需的宏定义
    add_definitions(
        -D_USE_MATH_DEFINES
        -DWIN32_LEAN_AND_MEAN
        -DNOMINMAX
        -D_CRT_SECURE_NO_WARNINGS
        -D_SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING
    )
    
    # 设置运行时库
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
    
    # 禁用特定警告
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4819")
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
    src/vector_math.cpp
    src/window.cpp
)

# 创建可执行文件
add_executable(SoftRenderer ${SOURCES})

# 复制资源文件
file(COPY assets DESTINATION ${CMAKE_BINARY_DIR}) 