cmake_minimum_required(VERSION 3.10)
project(SoftRenderer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 如果存在CUDA
find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    add_definitions(-DUSE_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CUDA_SOURCES
        src/ssaa_downsample.cu
        src/msaa_resolve.cu
        src/rasterizer_gpu.cu
        src/clear_gpu.cu
        src/gpu_vertex.cu)
endif()

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
    
    # 设置运行时库统一为动态 (/MD)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
    
    if(CUDAToolkit_FOUND)
        # 让 NVCC 使用共享版 cudart，并将 /MD 传递给 host 编译器
        set(CMAKE_CUDA_RUNTIME_LIBRARY "Shared")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/MD")
        set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -Xcompiler=/MDd")
    endif()
    
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

if(CUDAToolkit_FOUND)
    target_link_libraries(SoftRenderer PRIVATE CUDA::cudart)
endif()

# 复制资源文件
file(COPY assets DESTINATION ${CMAKE_BINARY_DIR}) 