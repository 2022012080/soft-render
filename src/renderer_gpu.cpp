#ifdef USE_CUDA
#include "renderer_gpu.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cstdint>

struct ColorDevice { unsigned char r, g, b, a; };
struct VertexDevice { float x, y, z; };
struct PixelDevice { ColorDevice color; float depth; };

extern "C" void launchRasterizeKernel(const void* d_vertices, const void* d_indices,
                                        int triCount, int width, int height,
                                        void* d_frameBuffer);

extern "C" void cudaClearFrameBuffer(void* buf, int count,
                                      unsigned char r, unsigned char g,
                                      unsigned char b, unsigned char a,
                                      float depth);

extern "C" void launchVertexShaderKernel(const void*, void*, int, const float*);

void RendererGPU::renderModel(const Model& model) {
    int w = getWidth();
    int h = getHeight();

    // ---------- 准备顶点 / 索引数据 ----------
    std::vector<VertexDevice> cpuVerts;
    std::vector<int> gpuIndices;
    size_t faceCount = model.getFaceCount();
    cpuVerts.reserve(faceCount * 3);
    gpuIndices.reserve(faceCount * 3);

    for (size_t i = 0; i < faceCount; ++i) {
        Vertex v0, v1, v2;
        model.getFaceVertices(static_cast<int>(i), v0, v1, v2);
        Renderer::ShaderVertex sv0 = vertexShader(v0);
        Renderer::ShaderVertex sv1 = vertexShader(v1);
        Renderer::ShaderVertex sv2 = vertexShader(v2);

        auto addSV = [&cpuVerts](const Renderer::ShaderVertex& sv){
            VertexDevice vd{ sv.worldPos.x, sv.worldPos.y, sv.worldPos.z };
            cpuVerts.push_back(vd);
        };
        addSV(sv0); addSV(sv1); addSV(sv2);
        int base = static_cast<int>(i * 3);
        gpuIndices.push_back(base);
        gpuIndices.push_back(base + 1);
        gpuIndices.push_back(base + 2);
    }

    // ---------- 设备内存分配 ----------
    VertexDevice* d_verts;
    int* d_idx;
    PixelDevice* d_frame;
    size_t vBytes = cpuVerts.size() * sizeof(VertexDevice);
    size_t iBytes = gpuIndices.size() * sizeof(int);
    size_t frameBytes = w * h * sizeof(PixelDevice);

    cudaMalloc(&d_verts, vBytes);
    cudaMalloc(&d_idx, iBytes);
    cudaMalloc(&d_frame, frameBytes);

    cudaMemcpy(d_verts, cpuVerts.data(), vBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_idx, gpuIndices.data(), iBytes, cudaMemcpyHostToDevice);

    // ---------- 顶点着色器 (GPU) ----------
    VertexDevice* d_clip;
    cudaMalloc(&d_clip, vBytes);

    Matrix4x4 mvpMat = viewportMatrix * projectionMatrix * viewMatrix * modelMatrix;
    float mvpArr[16];
    for(int r=0;r<4;++r)
        for(int c=0;c<4;++c)
            mvpArr[c*4+r]=mvpMat.m[r][c];

    launchVertexShaderKernel(d_verts, d_clip, static_cast<int>(cpuVerts.size()), mvpArr);

    // 后续 rasterizer 使用 clip 结果作为屏幕坐标
    cudaFree(d_verts);
    d_verts = d_clip;

    // ---------- 初始化帧缓冲 (clear) ----------
    Color black(0, 0, 0, 255);
    cudaClearFrameBuffer(getFrameBufferMutable().data(), w * h, black.r, black.g, black.b, black.a, 1.0f);
    cudaMemcpy(d_frame, getFrameBufferMutable().data(), frameBytes, cudaMemcpyHostToDevice);

    // ---------- 调度光栅化 ----------
    launchRasterizeKernel(d_verts, d_idx, static_cast<int>(faceCount), w, h, d_frame);

    // ---------- 拷回结果 ----------
    cudaMemcpy(getFrameBufferMutable().data(), d_frame, frameBytes, cudaMemcpyDeviceToHost);

    cudaFree(d_idx);
    cudaFree(d_frame);
}

#endif // USE_CUDA 