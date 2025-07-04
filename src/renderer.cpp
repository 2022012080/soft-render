#include "renderer.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <cmath>
#include <atomic>
#ifdef USE_CUDA
#include <cuda_runtime.h>
#ifdef __cplusplus
extern "C" {
#endif
void ssaaDownsampleKernelLauncher(const void* hi, void* lo,
                                         int lowW, int lowH, int scale, int highW);
#ifdef __cplusplus
}
#endif

void Renderer::downsampleFromHighResCUDA() {
    int hiSize = static_cast<int>(m_highResFrameBuffer.size());
    int loSize = static_cast<int>(frameBuffer.size());
    if (hiSize == 0 || loSize == 0) {
        downsampleFromHighRes();
        return;
    }

    size_t hiBytes = sizeof(Pixel) * hiSize;
    size_t loBytes = sizeof(Pixel) * loSize;

    Pixel* d_hi = nullptr;
    Pixel* d_lo = nullptr;
    cudaMalloc(&d_hi, hiBytes);
    cudaMalloc(&d_lo, loBytes);

    cudaMemcpy(d_hi, m_highResFrameBuffer.data(), hiBytes, cudaMemcpyHostToDevice);

    ssaaDownsampleKernelLauncher(d_hi, d_lo, width, height, m_ssaaScale, m_highResWidth);

    cudaMemcpy(frameBuffer.data(), d_lo, loBytes, cudaMemcpyDeviceToHost);

    cudaFree(d_hi);
    cudaFree(d_lo);
}

extern "C" void msaaResolveKernelLauncher(const void* sampleBuf, const void* bgColors,
                                            void* outPixels, int pixelCount, int sampleCount);

void Renderer::resolveMSAAToFrameBufferCUDA() {
    int pixelCount = width * height;
    int sampleCount = m_msaaSampleCount;
    if (pixelCount == 0 || sampleCount == 0) {
        return;
    }

    // ---------- 主机端展平数据 ----------
    struct ColorDevice { unsigned char r, g, b, a; };
    struct SampleDevice { ColorDevice color; float depth; unsigned char covered; };
    struct PixelDevice { ColorDevice color; float depth; };

    std::vector<SampleDevice> hostSamples;
    hostSamples.resize(static_cast<size_t>(pixelCount) * sampleCount);

    for (int idx = 0; idx < pixelCount; ++idx) {
        const auto& pixel = m_msaaFrameBuffer[idx];
        for (int s = 0; s < sampleCount; ++s) {
            const auto& samp = pixel.samples[s];
            SampleDevice d;
            d.color = { samp.color.r, samp.color.g, samp.color.b, samp.color.a };
            d.depth = samp.depth;
            d.covered = samp.covered ? 1 : 0;
            hostSamples[idx * sampleCount + s] = d;
        }
    }

    // 背景色数组
    std::vector<ColorDevice> hostBg(pixelCount);
    for (int i = 0; i < pixelCount; ++i) {
        hostBg[i] = { frameBuffer[i].color.r, frameBuffer[i].color.g,
                      frameBuffer[i].color.b, frameBuffer[i].color.a };
    }

    // 设备内存分配
    SampleDevice* d_samples = nullptr;
    ColorDevice* d_bg = nullptr;
    PixelDevice* d_out = nullptr;
    size_t samplesBytes = hostSamples.size() * sizeof(SampleDevice);
    size_t bgBytes = hostBg.size() * sizeof(ColorDevice);
    size_t outBytes = pixelCount * sizeof(PixelDevice);

    cudaMalloc(&d_samples, samplesBytes);
    cudaMalloc(&d_bg, bgBytes);
    cudaMalloc(&d_out, outBytes);

    cudaMemcpy(d_samples, hostSamples.data(), samplesBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bg, hostBg.data(), bgBytes, cudaMemcpyHostToDevice);

    // ---------- 调度 Kernel ----------
    msaaResolveKernelLauncher(d_samples, d_bg, d_out, pixelCount, sampleCount);

    // ---------- 结果拷回 ----------
    std::vector<PixelDevice> hostOut(pixelCount);
    cudaMemcpy(hostOut.data(), d_out, outBytes, cudaMemcpyDeviceToHost);

    // 写回到 frameBuffer 与 depthBuffer
    for (int i = 0; i < pixelCount; ++i) {
        frameBuffer[i].color = { hostOut[i].color.r, hostOut[i].color.g,
                                 hostOut[i].color.b, hostOut[i].color.a };
        frameBuffer[i].depth = hostOut[i].depth;
        if (i < depthBuffer.size()) depthBuffer[i] = hostOut[i].depth;
    }

    // 释放显存
    cudaFree(d_samples);
    cudaFree(d_bg);
    cudaFree(d_out);
}
#endif

// ================= ThreadPool 实现 =================
ThreadPool::ThreadPool(size_t numThreads) : stop(false), workingCount(0) {
    for (size_t i = 0; i < numThreads; ++i) {
        workers.emplace_back([this]() {
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(queueMutex);
                    condition.wait(lock, [this] { return stop || !tasks.empty(); });
                    if (stop && tasks.empty()) return;
                    task = std::move(tasks.front());
                    tasks.pop();
                    ++workingCount;
                }
                task();
                {
                    std::unique_lock<std::mutex> lock(queueMutex);
                    --workingCount;
                    if (tasks.empty() && workingCount == 0) {
                        doneCondition.notify_all();
                    }
                }
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        stop = true;
    }
    condition.notify_all();
    for (auto& worker : workers) {
        if (worker.joinable()) worker.join();
    }
}

void ThreadPool::enqueue(std::function<void()> task) {
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        tasks.push(std::move(task));
    }
    condition.notify_one();
}

void ThreadPool::waitAll() {
    std::unique_lock<std::mutex> lock(queueMutex);
    doneCondition.wait(lock, [this] { return tasks.empty() && workingCount == 0; });
}
// ================= ThreadPool 实现结束 =================

Renderer::Renderer(int w, int h) : width(w), height(h) {
    frameBuffer.resize(width * height);
    depthBuffer.resize(width * height);
    
    // 初始化超采样抗锯齿参数
    m_enableSSAA = false;
    m_ssaaScale = 4;
    m_highResWidth = 0;
    m_highResHeight = 0;
    
    // 初始化MSAA多重采样抗锯齿参数
    m_enableMSAA = false;
    m_msaaSampleCount = 4;
    m_msaaFrameBuffer.clear();
    m_samplePattern.clear();
    
    // 初始化多光源系统
    ambientIntensity = 0.2f;    // 环境光强度
    diffuseStrength = Vec3f(1.0f, 1.0f, 1.0f);     // 漫反射强度
    specularStrength = Vec3f(1.0f, 1.0f, 1.0f);    // 高光强度
    ambientStrength = Vec3f(1.0f, 1.0f, 1.0f);     // 环境光强度
    shininess = 32.0f;          // 新增：高光指数初始化
    
    // 添加默认光源
    lights.push_back(Light(Vec3f(3, 3, 3), Vec3f(1, 1, 1), 10.0f));  // 白色主光源
    lights.push_back(Light(Vec3f(-3, 2, 1), Vec3f(0.8, 0.6, 1.0), 5.0f));  // 紫色辅助光源
    lights.push_back(Light(Vec3f(0, -1, 0), Vec3f(1.0, 0.9, 0.8), 5.0f, LightType::DIRECTIONAL));  // 平面光源（向下）
    
    // 初始化绘制控制开关
    m_drawTriangleEdges = false;  // 默认关闭三角形描边
    m_drawLightRays = false;      // 默认关闭光线
    m_drawAxesAndGrid = true;     // 默认开启坐标轴和网格线
    
    // 初始化纹理启用控制
    m_enableTexture = false;      // 默认关闭贴图
    
    // 初始化法线贴图启用控制
    m_enableNormalMap = false;    // 默认关闭法线贴图
    
    // 新增：初始化 BRDF 模型参数
    m_enableBRDF = false;         // 默认关闭 BRDF
    m_roughness = 0.5f;           // 中等粗糙度
    m_metallic = 0.0f;            // 非金属材质
    m_F0 = Vec3f(0.04f, 0.04f, 0.04f);  // 非金属的基础反射率
    
    // 新增：初始化能量补偿参数
    m_enableEnergyCompensation = true;   // 默认启用能量补偿
    m_energyCompensationScale = 1.0f;    // 默认补偿强度
    m_lutInitialized = false;            // 查找表未初始化
    
    // 新增：初始化位移着色器参数
    m_enableDisplacement = false;        // 默认关闭位移着色器
    m_displacementScale = 0.5f;          // 默认位移强度
    m_displacementFrequency = 8.0f;      // 默认位移频率
    m_spineLength = 0.2f;                // 默认刺长度
    m_spineSharpness = 2.0f;             // 默认刺锐利度
    
    // 新增：初始化自发光参数
    m_enableEmission = false;            // 默认关闭自发光
    m_emissionStrength = 1.0f;           // 默认自发光强度
    m_emissionColor = Vec3f(1.0f, 1.0f, 1.0f);  // 默认白色自发光
    
    // 初始化法向量变换矩阵
    updateNormalMatrix();
    
    m_threadPool = std::make_unique<ThreadPool>(8);
}

void Renderer::enableSSAA(bool enable, int scale) {
    m_enableSSAA = enable;
    m_ssaaScale = scale;
    
    if (enable) {
        initializeSSAABuffers();
        std::cout << "SSAA enabled with " << scale << "x scale (" 
                  << m_highResWidth << "x" << m_highResHeight << " -> " 
                  << width << "x" << height << ")" << std::endl;
    } else {
        // 释放高分辨率缓冲区内存
        m_highResFrameBuffer.clear();
        m_highResDepthBuffer.clear();
        m_highResFrameBuffer.shrink_to_fit();
        m_highResDepthBuffer.shrink_to_fit();
        std::cout << "SSAA disabled" << std::endl;
    }
}

void Renderer::disableSSAA() {
    enableSSAA(false, m_ssaaScale);
}

// ================= MSAA 实现开始 =================

void Renderer::enableMSAA(bool enable, int sampleCount) {
    m_enableMSAA = enable;
    m_msaaSampleCount = sampleCount;
    
    if (enable) {
        initializeMSAABuffers();
        initializeSamplePattern();
        std::cout << "MSAA enabled with " << sampleCount << " samples per pixel" << std::endl;
    } else {
        // 释放MSAA缓冲区内存
        m_msaaFrameBuffer.clear();
        m_msaaFrameBuffer.shrink_to_fit();
        m_samplePattern.clear();
        std::cout << "MSAA disabled" << std::endl;
    }
}

void Renderer::disableMSAA() {
    enableMSAA(false, m_msaaSampleCount);
}

void Renderer::initializeMSAABuffers() {
    // 为每个像素分配MSAA采样点
    m_msaaFrameBuffer.resize(width * height);
    for (auto& pixel : m_msaaFrameBuffer) {
        pixel = MSAAPixel(m_msaaSampleCount);
    }
    
    std::cout << "Initialized MSAA buffers: " << width << "x" << height 
              << " with " << m_msaaSampleCount << " samples per pixel ("
              << (m_msaaFrameBuffer.size() * m_msaaSampleCount * sizeof(MSAASample)) / (1024*1024) 
              << " MB)" << std::endl;
}

void Renderer::initializeSamplePattern() {
    m_samplePattern.clear();
    
    if (m_msaaSampleCount == 4) {
        // 4x MSAA采样模式 (Rotated Grid)
        m_samplePattern.push_back(Vec2f(-0.25f, -0.25f));
        m_samplePattern.push_back(Vec2f(0.25f, -0.25f));
        m_samplePattern.push_back(Vec2f(-0.25f, 0.25f));
        m_samplePattern.push_back(Vec2f(0.25f, 0.25f));
    } else if (m_msaaSampleCount == 8) {
        // 8x MSAA采样模式 (8-Rook)
        m_samplePattern.push_back(Vec2f(-0.4375f, -0.1875f));
        m_samplePattern.push_back(Vec2f(-0.1875f, 0.0625f));
        m_samplePattern.push_back(Vec2f(0.0625f, -0.4375f));
        m_samplePattern.push_back(Vec2f(0.3125f, -0.3125f));
        m_samplePattern.push_back(Vec2f(-0.3125f, 0.1875f));
        m_samplePattern.push_back(Vec2f(-0.0625f, 0.4375f));
        m_samplePattern.push_back(Vec2f(0.1875f, -0.0625f));
        m_samplePattern.push_back(Vec2f(0.4375f, 0.3125f));
    } else {
        // 默认使用简单的采样模式
        for (int i = 0; i < m_msaaSampleCount; ++i) {
            float x = (i % 2) * 0.5f - 0.25f;
            float y = (i / 2) * 0.5f - 0.25f;
            m_samplePattern.push_back(Vec2f(x, y));
        }
    }
}

bool Renderer::isPointInTriangle(const Vec2f& p, const Vec2f& v0, const Vec2f& v1, const Vec2f& v2) {
    // 使用重心坐标判断点是否在三角形内
    float denom = (v1.x - v0.x) * (v2.y - v0.y) - (v2.x - v0.x) * (v1.y - v0.y);
    if (std::abs(denom) < 1e-6f) return false;
    
    float a = ((v1.x - p.x) * (v2.y - p.y) - (v2.x - p.x) * (v1.y - p.y)) / denom;
    float b = ((v2.x - p.x) * (v0.y - p.y) - (v0.x - p.x) * (v2.y - p.y)) / denom;
    float c = 1.0f - a - b;
    
    return (a >= 0.0f && b >= 0.0f && c >= 0.0f);
}

void Renderer::resolveMSAAToFrameBuffer() {
    if (!m_enableMSAA) return;
#ifdef USE_CUDA
    resolveMSAAToFrameBufferCUDA();
#else
    // -------- CPU 实现 --------
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int index = y * width + x;
            
            Color backgroundColor = frameBuffer[index].color;
            Color finalColor = m_msaaFrameBuffer[index].getFinalColor(backgroundColor);
            
            float minDepth = 1.0f;
            bool hasValidSample = false;
            for (const auto& sample : m_msaaFrameBuffer[index].samples) {
                if (sample.covered && sample.depth < minDepth) {
                    minDepth = sample.depth;
                    hasValidSample = true;
                }
            }
            
            frameBuffer[index].color = finalColor;
            if (hasValidSample) {
                frameBuffer[index].depth = minDepth;
                if (index < depthBuffer.size()) {
                    depthBuffer[index] = minDepth;
                }
            }
        }
    }
#endif
}

// ================= MSAA 实现结束 =================

void Renderer::initializeSSAABuffers() {
    m_highResWidth = width * m_ssaaScale;
    m_highResHeight = height * m_ssaaScale;
    
    // 分配高分辨率缓冲区
    m_highResFrameBuffer.resize(m_highResWidth * m_highResHeight);
    m_highResDepthBuffer.resize(m_highResWidth * m_highResHeight);
    
    std::cout << "Initialized SSAA buffers: " << m_highResWidth << "x" << m_highResHeight 
              << " (" << (m_highResFrameBuffer.size() * sizeof(Pixel) + 
                         m_highResDepthBuffer.size() * sizeof(float)) / (1024*1024) 
              << " MB)" << std::endl;
}

Matrix4x4 Renderer::createHighResViewportMatrix() const {
    if (!m_enableSSAA) {
        return viewportMatrix;
    }
    
    // 创建高分辨率视口变换矩阵
    float halfWidth = m_highResWidth * 0.5f;
    float halfHeight = m_highResHeight * 0.5f;
    
    Matrix4x4 highResViewport;
    highResViewport.m[0][0] = halfWidth;
    highResViewport.m[1][1] = halfHeight;   // 保持与原始视口矩阵一致，不翻转Y轴
    highResViewport.m[2][2] = 1.0f;         // 保持与原始视口矩阵一致
    highResViewport.m[0][3] = halfWidth;
    highResViewport.m[1][3] = halfHeight;
    
    return highResViewport;
}

void Renderer::clear(const Color& color) {
    // 清空常规帧缓冲
    for (auto& pixel : frameBuffer) {
        pixel.color = color;
        pixel.depth = 1.0f;
    }
    
    // 如果启用了SSAA，也清空高分辨率缓冲
    if (m_enableSSAA) {
        for (auto& pixel : m_highResFrameBuffer) {
            pixel.color = color;
            pixel.depth = 1.0f;
        }
    }
    
    // 如果启用了MSAA，也清空MSAA缓冲
    if (m_enableMSAA) {
        for (auto& pixel : m_msaaFrameBuffer) {
            for (auto& sample : pixel.samples) {
                sample.color = color;
                sample.depth = 1.0f;
                sample.covered = false;
            }
        }
    }
}

void Renderer::clearDepth() {
    // 清空常规深度缓冲
    for (auto& depth : depthBuffer) {
        depth = 1.0f;
    }
    
    // 如果启用了SSAA，也清空高分辨率深度缓冲
    if (m_enableSSAA) {
        for (auto& depth : m_highResDepthBuffer) {
            depth = 1.0f;
        }
    }
}

void Renderer::renderModel(const Model& model) {
    size_t faceCount = model.getFaceCount();
    if (faceCount == 0) return;

    if (m_enableSSAA) {
        Matrix4x4 originalViewport = viewportMatrix;
        viewportMatrix = createHighResViewportMatrix();

        for (size_t i = 0; i < faceCount; ++i) {
            m_threadPool->enqueue([this, &model, i]() {
                Vertex v0, v1, v2;
                model.getFaceVertices(static_cast<int>(i), v0, v1, v2);
                renderTriangleHighResWithFaceIdx(v0, v1, v2, static_cast<int>(i), &model);
            });
        }
        m_threadPool->waitAll();
        viewportMatrix = originalViewport;
    } else {
        // 原有路径：支持 MSAA/普通
    for (size_t i = 0; i < faceCount; ++i) {
        m_threadPool->enqueue([this, &model, i]() {
            Vertex v0, v1, v2;
            model.getFaceVertices(static_cast<int>(i), v0, v1, v2);
            renderTriangleWithFaceIdx(v0, v1, v2, static_cast<int>(i), &model);
        });
    }
    m_threadPool->waitAll();
    }
}

void Renderer::renderToHighRes(const Model& model) {
    // 保存原始视口矩阵
    Matrix4x4 originalViewport = viewportMatrix;
    
    // 设置高分辨率视口矩阵
    viewportMatrix = createHighResViewportMatrix();
    
    const auto& vertices = model.getVertices();
    
    for (size_t i = 0; i < vertices.size(); i += 3) {
        if (i + 2 < vertices.size()) {
            renderTriangleHighRes(vertices[i], vertices[i + 1], vertices[i + 2]);
        }
    }
    
    // 恢复原始视口矩阵
    viewportMatrix = originalViewport;
}

void Renderer::renderTriangleHighRes(const Vertex& v0, const Vertex& v1, const Vertex& v2) {
    ShaderVertex sv0 = vertexShader(v0);
    ShaderVertex sv1 = vertexShader(v1);
    ShaderVertex sv2 = vertexShader(v2);
    
    // Convert to view space for proper face culling
    Vec3f worldPos0 = modelMatrix * v0.position;
    Vec3f worldPos1 = modelMatrix * v1.position;
    Vec3f worldPos2 = modelMatrix * v2.position;
    Vec3f viewPos0 = viewMatrix * worldPos0;
    Vec3f viewPos1 = viewMatrix * worldPos1;
    Vec3f viewPos2 = viewMatrix * worldPos2;
    
    if (!isFrontFace(viewPos0, viewPos1, viewPos2)) {
        return;
    }
    
    // 渲染到高分辨率缓冲区
    rasterizeTriangleHighRes(sv0, sv1, sv2);
    
    // 绘制三角形边线（根据开关决定）- 在SSAA模式下也需要绘制
    if (m_drawTriangleEdges) {
        drawTriangleEdgesHighRes(sv0, sv1, sv2);
    }
}

void Renderer::rasterizeTriangleHighRes(const ShaderVertex& v0, const ShaderVertex& v1, const ShaderVertex& v2) {
    // 保持浮点数精度，避免提前转换为整数导致三角形退化
    float x0 = v0.position.x;
    float y0 = v0.position.y;
    float x1 = v1.position.x;
    float y1 = v1.position.y;
    float x2 = v2.position.x;
    float y2 = v2.position.y;
    
    // 计算边界框
    int minX = std::max(0, static_cast<int>(std::floor(std::min({x0, x1, x2}))));
    int maxX = std::min(m_highResWidth - 1, static_cast<int>(std::ceil(std::max({x0, x1, x2}))));
    int minY = std::max(0, static_cast<int>(std::floor(std::min({y0, y1, y2}))));
    int maxY = std::min(m_highResHeight - 1, static_cast<int>(std::ceil(std::max({y0, y1, y2}))));

    for (int y = minY; y <= maxY; y++) {
        for (int x = minX; x <= maxX; x++) {
            // 使用浮点数坐标进行重心坐标计算，保持精度
            Vec3f bary = barycentric(Vec2f(x0, y0), 
                                   Vec2f(x1, y1), 
                                   Vec2f(x2, y2), 
                                   Vec2f(static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f));
            
            if (bary.x >= 0 && bary.y >= 0 && bary.z >= 0) {
                float depth = bary.x * v0.position.z + bary.y * v1.position.z + bary.z * v2.position.z;
                
                if (depthTestHighRes(x, y, depth)) {
                    ShaderFragment fragment;
                    // 使用正确的重心坐标插值
                    fragment.position = v0.position * bary.x + v1.position * bary.y + v2.position * bary.z;
                    fragment.normal = v0.normal * bary.x + v1.normal * bary.y + v2.normal * bary.z;
                    fragment.texCoord = v0.texCoord * bary.x + v1.texCoord * bary.y + v2.texCoord * bary.z;
                    fragment.worldPos = v0.worldPos * bary.x + v1.worldPos * bary.y + v2.worldPos * bary.z;
                    fragment.localPos = v0.localPos * bary.x + v1.localPos * bary.y + v2.localPos * bary.z;
                    fragment.localNormal = v0.localNormal * bary.x + v1.localNormal * bary.y + v2.localNormal * bary.z;
                    
                    fragment.normal = fragment.normal.normalize();
                    fragment.localNormal = fragment.localNormal.normalize();
                    
                    Color color = fragmentShader(fragment);
                    setPixelHighRes(x, y, color, depth);
                }
            }
        }
    }
}

bool Renderer::depthTestHighRes(int x, int y, float depth) {
    int index = y * m_highResWidth + x;
    if (index >= 0 && index < static_cast<int>(m_highResDepthBuffer.size()) && depth < m_highResDepthBuffer[index]) {
        m_highResDepthBuffer[index] = depth;
        return true;
    }
    return false;
}

void Renderer::setPixelHighRes(int x, int y, const Color& color, float depth) {
    if (x < 0 || x >= m_highResWidth || y < 0 || y >= m_highResHeight) {
        return;
    }
    
    int index = y * m_highResWidth + x;
    if (index >= 0 && index < static_cast<int>(m_highResFrameBuffer.size())) {
        m_highResFrameBuffer[index].color = color;
        m_highResFrameBuffer[index].depth = depth;
    }
}

void Renderer::downsampleFromHighRes() {
    if (!m_enableSSAA || m_highResFrameBuffer.empty()) {
        return;
    }
    
    // 对于每个低分辨率像素，从对应的高分辨率区域采样并求平均值
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float totalR = 0.0f, totalG = 0.0f, totalB = 0.0f;
            int sampleCount = 0;
            
            // 计算对应的高分辨率区域
            int startX = x * m_ssaaScale;
            int startY = y * m_ssaaScale;
            int endX = startX + m_ssaaScale;
            int endY = startY + m_ssaaScale;
            
            // 确保不超出边界
            endX = std::min(endX, m_highResWidth);
            endY = std::min(endY, m_highResHeight);
            
            // 对高分辨率区域内的所有像素求平均
            for (int highY = startY; highY < endY; highY++) {
                for (int highX = startX; highX < endX; highX++) {
                    int highIndex = highY * m_highResWidth + highX;
                    if (highIndex >= 0 && highIndex < static_cast<int>(m_highResFrameBuffer.size())) {
                        const Color& highResColor = m_highResFrameBuffer[highIndex].color;
                        totalR += highResColor.r;
                        totalG += highResColor.g;
                        totalB += highResColor.b;
                        sampleCount++;
                    }
                }
            }
            
            // 计算平均颜色
            if (sampleCount > 0) {
                Color avgColor(
                    static_cast<unsigned char>(totalR / sampleCount),
                    static_cast<unsigned char>(totalG / sampleCount),
                    static_cast<unsigned char>(totalB / sampleCount)
                );
                
                int lowIndex = y * width + x;
                if (lowIndex >= 0 && lowIndex < static_cast<int>(frameBuffer.size())) {
                    frameBuffer[lowIndex].color = avgColor;
                    // 深度值使用中心点的深度
                    int centerHighX = startX + m_ssaaScale / 2;
                    int centerHighY = startY + m_ssaaScale / 2;
                    int centerHighIndex = centerHighY * m_highResWidth + centerHighX;
                    if (centerHighIndex >= 0 && centerHighIndex < static_cast<int>(m_highResFrameBuffer.size())) {
                        frameBuffer[lowIndex].depth = m_highResFrameBuffer[centerHighIndex].depth;
                    }
                }
            }
        }
    }
}

void Renderer::renderTriangleWithFaceIdx(const Vertex& v0, const Vertex& v1, const Vertex& v2, int faceIdx, const Model* pModel) {
    ShaderVertex sv0 = vertexShader(v0);
    ShaderVertex sv1 = vertexShader(v1);
    ShaderVertex sv2 = vertexShader(v2);
    
    // Convert to view space for proper face culling
    Vec3f worldPos0 = modelMatrix * v0.position;
    Vec3f worldPos1 = modelMatrix * v1.position;
    Vec3f worldPos2 = modelMatrix * v2.position;
    Vec3f viewPos0 = viewMatrix * worldPos0;
    Vec3f viewPos1 = viewMatrix * worldPos1;
    Vec3f viewPos2 = viewMatrix * worldPos2;
    
    if (!isFrontFace(viewPos0, viewPos1, viewPos2)) return;
    
    // 根据抗锯齿模式选择渲染方法
    if (m_enableMSAA) {
        rasterizeTriangleMSAA(sv0, sv1, sv2, faceIdx, pModel);
    } else {
        rasterizeTriangleWithFaceIdx(sv0, sv1, sv2, faceIdx, pModel);
    }
    
    if (m_drawTriangleEdges) {
        drawTriangleEdges(sv0, sv1, sv2);
    }
}

Renderer::ShaderVertex Renderer::vertexShader(const Vertex& vertex) {
    ShaderVertex sv;
    
    // 保存本地坐标信息
    sv.localPos = vertex.position;
    sv.localNormal = vertex.normal;
    
    // 应用位移
    Vec3f displacedPosition = vertex.position;
    if (m_enableDisplacement) {
        Vec3f displacement = calculateSeaUrchinDisplacement(vertex.position, vertex.normal);
        displacedPosition = displacedPosition + displacement;
        
        // 更新法线（简化处理，实际应用中可能需要更复杂的法线计算）
        sv.localNormal = (vertex.normal + displacement.normalize() * 0.5f).normalize();
    }
    
    // 应用变换
    sv.worldPos = modelMatrix * displacedPosition;
    Vec3f viewPos = viewMatrix * sv.worldPos;
    Vec3f clipPos = projectionMatrix * viewPos;
    sv.position = viewportMatrix * clipPos;
    
    // 变换法向量 - 使用法向量变换矩阵
    sv.normal = (normalMatrix * Vec4f(sv.localNormal, 0.0f)).xyz().normalize();
    sv.texCoord = vertex.texCoord;
    
    return sv;
}

Color Renderer::fragmentShader(const ShaderFragment& fragment) {
    // 设置基础颜色 - 如果启用纹理且有纹理则使用纹理，否则使用白色
    Vec3f baseColor(1, 1, 1);  // 默认白色
    
    // 根据纹理启用开关决定是否进行纹理采样
    if (m_enableTexture && currentTexture && currentTexture->isValid()) {
        baseColor = currentTexture->sampleVec3f(fragment.texCoord.x, fragment.texCoord.y);
    }
    
    // 处理法线贴图
    Vec3f normal = fragment.localNormal;
    if (m_enableNormalMap && currentNormalMap && currentNormalMap->isValid()) {
        // 从法线贴图采样法线（RGB值在0-255范围内）
        Vec3f normalMapSample = currentNormalMap->sampleVec3f(fragment.texCoord.x, fragment.texCoord.y);
        
        // 将RGB值从[0,1]范围转换到[-1,1]范围
        Vec3f tangentSpaceNormal = normalMapSample * 2.0f - Vec3f(1.0f, 1.0f, 1.0f);
        
        // 简化的法线扰动：直接在本地坐标系中应用
        Vec3f N = fragment.localNormal.normalize();
        
        // 切线空间
        Vec3f up = (std::abs(N.y) < 0.9f) ? Vec3f(0, 1, 0) : Vec3f(1, 0, 0);
        Vec3f T = up.cross(N).normalize();  // 切线
        Vec3f B = N.cross(T);               // 副切线
        
        // 将切线空间的法线转换到本地坐标系
        normal = T * tangentSpaceNormal.x + B * tangentSpaceNormal.y + N * tangentSpaceNormal.z;
        normal = normal.normalize();
    }
    
    Vec3f finalColor;
    
    // 根据是否启用法线贴图来选择光照模型
    if (!m_enableNormalMap && m_enableBRDF) {
        // 当不使用法线贴图时，启用 BRDF 模型
        finalColor = calculateBRDFLighting(fragment.localPos, normal, baseColor);
    } else {
        // 使用传统 Phong 光照模型
        finalColor = calculateLighting(fragment.localPos, normal, baseColor);
    }
    
    // 添加自发光计算
    if (m_enableEmission) {
        // 将自发光颜色与强度相乘，然后添加到最终颜色中
        Vec3f emission = m_emissionColor * m_emissionStrength;
        finalColor = finalColor + emission * baseColor;  // 使用基础颜色调制自发光
    }
    
    return Color::fromVec3f(finalColor);
}

void Renderer::rasterizeTriangle(const ShaderVertex& v0, const ShaderVertex& v1, const ShaderVertex& v2) {
    // 保持浮点数精度，避免提前转换为整数导致三角形退化
    float x0 = v0.position.x;
    float y0 = v0.position.y;
    float x1 = v1.position.x;
    float y1 = v1.position.y;
    float x2 = v2.position.x;
    float y2 = v2.position.y;
    
    // 计算边界框（使用浮点数）
    int minX = std::max(0, static_cast<int>(std::floor(std::min({x0, x1, x2}))));
    int maxX = std::min(width - 1, static_cast<int>(std::ceil(std::max({x0, x1, x2}))));
    int minY = std::max(0, static_cast<int>(std::floor(std::min({y0, y1, y2}))));
    int maxY = std::min(height - 1, static_cast<int>(std::ceil(std::max({y0, y1, y2}))));
   
    for (int y = minY; y <= maxY; y++) {
        for (int x = minX; x <= maxX; x++) {
            // 使用浮点数坐标进行重心坐标计算，保持精度
            Vec3f bary = barycentric(Vec2f(x0, y0), 
                                   Vec2f(x1, y1), 
                                   Vec2f(x2, y2), 
                                   Vec2f(static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f));
            
            if (bary.x >= 0 && bary.y >= 0 && bary.z >= 0) {
                float depth = bary.x * v0.position.z + bary.y * v1.position.z + bary.z * v2.position.z;
                
                if (depthTest(x, y, depth)) {
                    ShaderFragment fragment;
                    // 使用正确的重心坐标插值
                    fragment.position = v0.position * bary.x + v1.position * bary.y + v2.position * bary.z;
                    fragment.normal = v0.normal * bary.x + v1.normal * bary.y + v2.normal * bary.z;
                    fragment.texCoord = v0.texCoord * bary.x + v1.texCoord * bary.y + v2.texCoord * bary.z;
                    fragment.worldPos = v0.worldPos * bary.x + v1.worldPos * bary.y + v2.worldPos * bary.z;
                    fragment.localPos = v0.localPos * bary.x + v1.localPos * bary.y + v2.localPos * bary.z;
                    fragment.localNormal = v0.localNormal * bary.x + v1.localNormal * bary.y + v2.localNormal * bary.z;
                    
                    fragment.normal = fragment.normal.normalize();
                    fragment.localNormal = fragment.localNormal.normalize();
                    
                    Color color = fragmentShader(fragment);
                    setPixel(x, y, color, depth);
                }
            }
        }
    }
}

Vec3f Renderer::barycentric(const Vec2f& a, const Vec2f& b, const Vec2f& c, const Vec2f& p) {
    Vec2f v0 = b - a;
    Vec2f v1 = c - a;
    Vec2f v2 = p - a;
    
    float d00 = v0.dot(v0);
    float d01 = v0.dot(v1);
    float d11 = v1.dot(v1);
    float d20 = v2.dot(v0);
    float d21 = v2.dot(v1);
    
    float denom = d00 * d11 - d01 * d01;
    
    
    float v = (d11 * d20 - d01 * d21) / denom;
    float w = (d00 * d21 - d01 * d20) / denom;
    float u = 1.0f - v - w;
    
    return Vec3f(u, v, w);
}

bool Renderer::depthTest(int x, int y, float depth) {
    int index = y * width + x;
    if (depth < depthBuffer[index]) {
        depthBuffer[index] = depth;
        return true;
    }
    return false;
}

void Renderer::setPixel(int x, int y, const Color& color, float depth) {
    if (x < 0 || x >= width || y < 0 || y >= height) {
        return;
    }
    
    int index = y * width + x;
    frameBuffer[index].color = color;
    frameBuffer[index].depth = depth;
}

Vec3f Renderer::calculateLighting(const Vec3f& localPos, const Vec3f& localNormal, const Vec3f& baseColor) {
    // 环境光 - 使用环境光强度系数
    Vec3f ambient = baseColor * (ambientIntensity * ambientStrength);
    
    // 初始化总光照贡献
    Vec3f totalLighting = ambient;
    
    // 计算所有光源的贡献
    for (const auto& light : lights) {
        if (light.enabled) {
            totalLighting += calculateSingleLight(light, localPos, localNormal, baseColor);
        }
    }
    
    return totalLighting;
}

Vec3f Renderer::calculateSingleLight(const Light& light, const Vec3f& localPos, const Vec3f& localNormal, const Vec3f& baseColor) {
    Vec3f lightDir;
    float attenuation;
    
    if (light.type == LightType::DIRECTIONAL) {
        // 平面光源：使用固定方向，无距离衰减
        lightDir = -light.position.normalize();  // 光源position作为方向向量，取反得到光照方向
        attenuation = light.intensity;  // 平面光源无距离衰减
    } else {
        // 点光源：原有逻辑
        Vec3f lightVector = light.position - localPos;
        float distance = lightVector.length();
        lightDir = lightVector.normalize();
        
        // 计算标准距离衰减（I/r²）
        float r_squared = distance * distance;
        attenuation = light.intensity / r_squared;
    }
    
    // 漫反射光照（Lambert模型） - 使用本地坐标系的法向量
    float diffuseIntensity = std::max(0.0f, localNormal.dot(lightDir));
    Vec3f diffuse = baseColor * light.color * diffuseIntensity * attenuation * this->diffuseStrength;
    
    // 高光计算 - 只有当表面面向光源时才计算
    Vec3f specular(0, 0, 0);
    if (diffuseIntensity > 0.0f) {  // 只有面向光源的表面才有高光
        // 正确的视角方向计算：直接将摄像机位置变换到物体本地坐标系
        // 1. 获取摄像机在世界坐标系的位置
        Vec3f worldCameraPos = VectorMath::inverse(viewMatrix) * Vec3f(0, 0, 0);
        
        // 2. 将摄像机位置变换到物体本地坐标系
        Matrix4x4 invModelMatrix = VectorMath::inverse(modelMatrix);
        Vec3f localCameraPos = invModelMatrix * worldCameraPos;
        
        // 3. 在本地坐标系中计算视角方向（从表面点指向摄像机）
        Vec3f localViewDir = (localCameraPos - localPos).normalize();
        
        // 镜面反射光照（Phong模型）
        Vec3f reflectDir = localNormal * (2.0f * localNormal.dot(lightDir)) - lightDir;
        reflectDir = reflectDir.normalize();
        
        // 使用可调节的高光指数
        float specularIntensity = std::pow(std::max(0.0f, localViewDir.dot(reflectDir)), this->shininess);
        
        // 增加镜面反射系数
        float Ks = 0.5f;  // 从0.5增加到1.0
        specular = light.color * Ks * attenuation * specularIntensity * this->specularStrength;
    }
    
    return diffuse + specular;
}

bool Renderer::isFrontFace(const Vec3f& v0, const Vec3f& v1, const Vec3f& v2) {
    Vec3f edge1 = v1 - v0;
    Vec3f edge2 = v2 - v0;
    Vec3f normal = edge1.cross(edge2);
    
    // In view space, camera looks down negative Z axis
    // Front faces have normals pointing towards camera (negative Z)
    return normal.z < 0;
}

const std::vector<Color>& Renderer::getColorBuffer() const {
    static std::vector<Color> colorBuffer;
    colorBuffer.clear();
    colorBuffer.reserve(frameBuffer.size());
    
    for (const auto& pixel : frameBuffer) {
        colorBuffer.push_back(pixel.color);
    }
    
    return colorBuffer;
}

bool Renderer::saveImage(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Cannot create output file: " << filename << std::endl;
        return false;
    }
    
    // BMP文件头
    int fileSize = 54 + width * height * 3;
    int dataOffset = 54;
    
    // 写入BMP文件头
    file.write("BM", 2);
    file.write(reinterpret_cast<const char*>(&fileSize), 4);
    file.write("\0\0\0\0", 4);
    file.write(reinterpret_cast<const char*>(&dataOffset), 4);
    
    // 信息头
    int infoHeaderSize = 40;
    file.write(reinterpret_cast<const char*>(&infoHeaderSize), 4);
    file.write(reinterpret_cast<const char*>(&width), 4);
    file.write(reinterpret_cast<const char*>(&height), 4);
    
    short planes = 1;
    short bitsPerPixel = 24;
    file.write(reinterpret_cast<const char*>(&planes), 2);
    file.write(reinterpret_cast<const char*>(&bitsPerPixel), 2);
    
    int compression = 0;
    int imageSize = width * height * 3;
    file.write(reinterpret_cast<const char*>(&compression), 4);
    file.write(reinterpret_cast<const char*>(&imageSize), 4);
    
    int xPixelsPerM = 0;
    int yPixelsPerM = 0;
    file.write(reinterpret_cast<const char*>(&xPixelsPerM), 4);
    file.write(reinterpret_cast<const char*>(&yPixelsPerM), 4);
    
    int colorsUsed = 0;
    int importantColors = 0;
    file.write(reinterpret_cast<const char*>(&colorsUsed), 4);
    file.write(reinterpret_cast<const char*>(&importantColors), 4);
    
    // 写入像素数据（BMP是倒置的）
    for (int y = height - 1; y >= 0; y--) {
        for (int x = 0; x < width; x++) {
            const Color& color = frameBuffer[y * width + x].color;
            file.write(reinterpret_cast<const char*>(&color.b), 1);
            file.write(reinterpret_cast<const char*>(&color.g), 1);
            file.write(reinterpret_cast<const char*>(&color.r), 1);
        }
        
        // 行对齐
        int padding = (4 - (width * 3) % 4) % 4;
        for (int i = 0; i < padding; i++) {
            file.write("\0", 1);
        }
    }
    
    file.close();
    return true;
}

void Renderer::updateNormalMatrix() {
    // 法向量变换矩阵是模型矩阵的逆转置矩阵
    // 对于仅包含旋转和均匀缩放的变换，可以直接使用模型矩阵的3x3部分
    normalMatrix = VectorMath::transpose(VectorMath::inverse(modelMatrix));
}

// 绘制线段 - 使用改进的Bresenham算法
void Renderer::drawLine(const Vec3f& start, const Vec3f& end, const Color& color, float width) {
    // 将3D点转换到屏幕坐标
    Vec3f worldStart = modelMatrix * start;
    Vec3f worldEnd = modelMatrix * end;
    Vec3f viewStart = viewMatrix * worldStart;
    Vec3f viewEnd = viewMatrix * worldEnd;
    Vec3f clipStart = projectionMatrix * viewStart;
    Vec3f clipEnd = projectionMatrix * viewEnd;
    Vec3f screenStart = viewportMatrix * clipStart;
    Vec3f screenEnd = viewportMatrix * clipEnd;
    
    int x0 = static_cast<int>(screenStart.x);
    int y0 = static_cast<int>(screenStart.y);
    int x1 = static_cast<int>(screenEnd.x);
    int y1 = static_cast<int>(screenEnd.y);
    float z0 = screenStart.z;
    float z1 = screenEnd.z;
    
    // 改进的Bresenham直线算法
    int dx = abs(x1 - x0);
    int dy = abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;
    int err = dx - dy;
    
    int x = x0, y = y0;
    float totalDistance = std::sqrt(static_cast<float>(dx * dx + dy * dy));
    
    // 确保线宽至少为1
    int lineWidth = std::max(1, static_cast<int>(width));
    int halfWidth = lineWidth / 2;
    
    while (true) {
        // 计算当前点的深度值
        float currentDistance = std::sqrt(static_cast<float>((x - x0) * (x - x0) + (y - y0) * (y - y0)));
        float t = (totalDistance > 0) ? (currentDistance / totalDistance) : 0;
        float z = VectorMath::lerp(z0, z1, t);
        
        // 绘制像素（考虑线宽）- 使用圆形笔刷
        for (int dx_offset = -halfWidth; dx_offset <= halfWidth; dx_offset++) {
            for (int dy_offset = -halfWidth; dy_offset <= halfWidth; dy_offset++) {
                // 使用圆形笔刷，避免方形笔刷的锯齿
                float distance = std::sqrt(static_cast<float>(dx_offset * dx_offset + dy_offset * dy_offset));
                if (distance <= halfWidth) {
                    int px = x + dx_offset;
                    int py = y + dy_offset;
                    if (px >= 0 && px < this->width && py >= 0 && py < this->height) {
                        if (depthTest(px, py, z)) {
                            // 如果颜色有alpha通道，进行alpha混合
                            if (color.a < 255) {
                                int index = py * this->width + px;
                                Color existingColor = frameBuffer[index].color;
                                
                                float alpha = color.a / 255.0f;
                                Color blendedColor(
                                    static_cast<unsigned char>(color.r * alpha + existingColor.r * (1.0f - alpha)),
                                    static_cast<unsigned char>(color.g * alpha + existingColor.g * (1.0f - alpha)),
                                    static_cast<unsigned char>(color.b * alpha + existingColor.b * (1.0f - alpha)),
                                    255
                                );
                                setPixel(px, py, blendedColor, z);
                            } else {
                                setPixel(px, py, color, z);
                            }
                        }
                    }
                }
            }
        }
        
        if (x == x1 && y == y1) break;
        
        int e2 = 2 * err;
        if (e2 > -dy) {
            err -= dy;
            x += sx;
        }
        if (e2 < dx) {
            err += dx;
            y += sy;
        }
    }
}

// 绘制线段到高分辨率缓冲区 - 用于SSAA模式
void Renderer::drawLineHighRes(const Vec3f& start, const Vec3f& end, const Color& color, float width) {
    // 将3D点转换到屏幕坐标（使用高分辨率视口矩阵）
    Vec3f worldStart = modelMatrix * start;
    Vec3f worldEnd = modelMatrix * end;
    Vec3f viewStart = viewMatrix * worldStart;
    Vec3f viewEnd = viewMatrix * worldEnd;
    Vec3f clipStart = projectionMatrix * viewStart;
    Vec3f clipEnd = projectionMatrix * viewEnd;
    
    // 使用高分辨率视口矩阵进行变换
    Matrix4x4 highResViewport = createHighResViewportMatrix();
    Vec3f screenStart = highResViewport * clipStart;
    Vec3f screenEnd = highResViewport * clipEnd;
    
    int x0 = static_cast<int>(screenStart.x);
    int y0 = static_cast<int>(screenStart.y);
    int x1 = static_cast<int>(screenEnd.x);
    int y1 = static_cast<int>(screenEnd.y);
    float z0 = screenStart.z;
    float z1 = screenEnd.z;
    
    // 改进的Bresenham直线算法
    int dx = abs(x1 - x0);
    int dy = abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;
    int err = dx - dy;
    
    int x = x0, y = y0;
    float totalDistance = std::sqrt(static_cast<float>(dx * dx + dy * dy));
    
    // 确保线宽至少为1
    int lineWidth = std::max(1, static_cast<int>(width));
    int halfWidth = lineWidth / 2;
    
    while (true) {
        // 计算当前点的深度值
        float currentDistance = std::sqrt(static_cast<float>((x - x0) * (x - x0) + (y - y0) * (y - y0)));
        float t = (totalDistance > 0) ? (currentDistance / totalDistance) : 0;
        float z = VectorMath::lerp(z0, z1, t);
        
        // 绘制像素（考虑线宽）- 使用圆形笔刷
        for (int dx_offset = -halfWidth; dx_offset <= halfWidth; dx_offset++) {
            for (int dy_offset = -halfWidth; dy_offset <= halfWidth; dy_offset++) {
                // 使用圆形笔刷，避免方形笔刷的锯齿
                float distance = std::sqrt(static_cast<float>(dx_offset * dx_offset + dy_offset * dy_offset));
                if (distance <= halfWidth) {
                    int px = x + dx_offset;
                    int py = y + dy_offset;
                    if (px >= 0 && px < m_highResWidth && py >= 0 && py < m_highResHeight) {
                        if (depthTestHighRes(px, py, z)) {
                            // 如果颜色有alpha通道，进行alpha混合
                            if (color.a < 255) {
                                int index = py * m_highResWidth + px;
                                Color existingColor = m_highResFrameBuffer[index].color;
                                
                                float alpha = color.a / 255.0f;
                                Color blendedColor(
                                    static_cast<unsigned char>(color.r * alpha + existingColor.r * (1.0f - alpha)),
                                    static_cast<unsigned char>(color.g * alpha + existingColor.g * (1.0f - alpha)),
                                    static_cast<unsigned char>(color.b * alpha + existingColor.b * (1.0f - alpha)),
                                    255
                                );
                                setPixelHighRes(px, py, blendedColor, z);
                            } else {
                                setPixelHighRes(px, py, color, z);
                            }
                        }
                    }
                }
            }
        }
        
        if (x == x1 && y == y1) break;
        
        int e2 = 2 * err;
        if (e2 > -dy) {
            err -= dy;
            x += sx;
        }
        if (e2 < dx) {
            err += dx;
            y += sy;
        }
    }
}

// 绘制坐标轴
void Renderer::drawAxes(float maxLength) {
    maxLength = 30.0f;
    Vec3f origin(0, 0, 0);
    int screenW = m_enableSSAA ? m_highResWidth : width;
    int screenH = m_enableSSAA ? m_highResHeight : height;
    Matrix4x4 vpMat = m_enableSSAA ? createHighResViewportMatrix() : viewportMatrix;

    // 计算原点在屏幕空间的位置
    Vec3f worldOrigin = modelMatrix * origin;
    Vec3f viewOrigin = viewMatrix * worldOrigin;
    Vec3f clipOrigin = projectionMatrix * viewOrigin;
    Vec3f screenOrigin = vpMat * clipOrigin;

    // 对每个轴进行处理
    for (int axis = 0; axis < 3; axis++) {
        Vec3f direction(0, 0, 0);
        Color axisColor;
        
        // 设置轴的方向和颜色
        switch(axis) {
            case 0: // X轴
                direction = Vec3f(1, 0, 0);
                axisColor = Color(255, 0, 0);
                break;
            case 1: // Z轴
                direction = Vec3f(0, 1, 0);
                axisColor = Color(0, 255, 0);
                break;
            case 2: // Y轴
                direction = Vec3f(0, 0, 1);
                axisColor = Color(0, 0, 255);
                break;
        }

        // 在两个方向上延伸轴线
        for (int sign = -1; sign <= 1; sign += 2) {
            float currentLength = 0;
            bool foundStart = false;
            Vec3f startPoint = origin;
            Vec3f lastValidPoint = origin;
            bool wasInScreen = false;
            
            // 以较小的步长搜索可见部分
            while (true) {
                Vec3f endPoint = origin + direction * (currentLength * sign);
                Vec3f worldEnd = modelMatrix * endPoint;
                Vec3f viewEnd = viewMatrix * worldEnd;
                Vec3f clipEnd = projectionMatrix * viewEnd;
                Vec3f screenEnd = vpMat * clipEnd;

                // 检查点是否在屏幕内
                bool inScreen = screenEnd.x >= 0 && screenEnd.x < screenW && 
                              screenEnd.y >= 0 && screenEnd.y < screenH;

                if (inScreen) {
                    if (!foundStart) {
                        // 找到起始点
                        startPoint = endPoint;
                        foundStart = true;
                    }
                    lastValidPoint = endPoint;
                    wasInScreen = true;
                }
                else if (wasInScreen) {
                    // 找到结束点，绘制这段轴线
                    if (m_enableSSAA) {
                        drawLineHighRes(startPoint, lastValidPoint, axisColor, 2.0f * m_ssaaScale);
                    } else {
                        drawLine(startPoint, lastValidPoint, axisColor, 2.0f);
                    }
                    break;
                }
                
                // 如果已经搜索了很远的距离还没找到屏幕边界，就停止搜索
                if (currentLength > 1000.0f) {
                    if (wasInScreen) {
                        // 如果之前找到过屏幕内的点，绘制到最后一个有效点
                        if (m_enableSSAA) {
                            drawLineHighRes(startPoint, lastValidPoint, axisColor, 2.0f * m_ssaaScale);
                        } else {
                            drawLine(startPoint, lastValidPoint, axisColor, 2.0f);
                        }
                    }
                    break;
                }
                
                currentLength += 0.01f; // 使用更大的步长以提高性能
            }
        }
    }
}

// 绘制网格
void Renderer::drawGrid(float size, int divisions) {
    int screenW = m_enableSSAA ? m_highResWidth : width;
    int screenH = m_enableSSAA ? m_highResHeight : height;
    Matrix4x4 vpMat = m_enableSSAA ? createHighResViewportMatrix() : viewportMatrix;
    Color gridColor(128, 128, 128, 128); // 半透明灰色

    // 在X方向和Z方向上绘制网格线
    for (int dir = 0; dir < 2; dir++) {
        Vec3f direction = (dir == 0) ? Vec3f(1, 0, 0) : Vec3f(0, 1, 0); // X方向或Z方向
        Vec3f perpDirection = (dir == 0) ? Vec3f(0, 1, 0) : Vec3f(1, 0, 0); // 垂直于当前方向

        // 对每条网格线
        for (int i = -30; i <= 30; i++) {
            Vec3f lineBase = perpDirection * static_cast<float>(i);

            // 在两个方向上延伸网格线
            float currentLength = 0;
            bool foundStart = false;
            Vec3f startPoint = lineBase;
            Vec3f lastValidPoint = lineBase;
            bool wasInScreen = false;
            bool hasDrawn = false;
            Vec3f lastPoint = lineBase;

            // 向一个方向搜索
            while (true) {
                Vec3f endPoint = lineBase + direction * currentLength;
                Vec3f worldEnd = modelMatrix * endPoint;
                Vec3f viewEnd = viewMatrix * worldEnd;
                Vec3f clipEnd = projectionMatrix * viewEnd;
                Vec3f screenEnd = vpMat * clipEnd;

                bool inScreen = screenEnd.x >= 0 && screenEnd.x < screenW && 
                              screenEnd.y >= 0 && screenEnd.y < screenH;

                if (inScreen) {
                    if (!foundStart) {
                        // 如果这是第一个可见点，且之前有不可见点，绘制到这个点
                        if (currentLength > 0) {
                            startPoint = lastPoint;
                            if (m_enableSSAA) {
                                drawLineHighRes(startPoint, endPoint, gridColor, 1.0f * m_ssaaScale);
                            } else {
                                drawLine(startPoint, endPoint, gridColor, 1.0f);
                            }
                        }
                        foundStart = true;
                    }
                    lastValidPoint = endPoint;
                    wasInScreen = true;
                }
                else if (wasInScreen && !hasDrawn) {
                    // 找到结束点，绘制这段网格线
                    if (m_enableSSAA) {
                        drawLineHighRes(startPoint, endPoint, gridColor, 1.0f * m_ssaaScale);
                    } else {
                        drawLine(startPoint, endPoint, gridColor, 1.0f);
                    }
                    hasDrawn = true;
                    break;
                }

                if (currentLength > 1000.0f) {
                    if (wasInScreen && !hasDrawn) {
                        if (m_enableSSAA) {
                            drawLineHighRes(startPoint, endPoint, gridColor, 1.0f * m_ssaaScale);
                        } else {
                            drawLine(startPoint, endPoint, gridColor, 1.0f);
                        }
                    }
                    break;
                }

                lastPoint = endPoint;
                currentLength += 1.0f;
            }

            // 向另一个方向搜索
            currentLength = 0;
            foundStart = false;
            startPoint = lineBase;
            lastValidPoint = lineBase;
            wasInScreen = false;
            hasDrawn = false;
            lastPoint = lineBase;

            while (true) {
                Vec3f endPoint = lineBase - direction * currentLength;
                Vec3f worldEnd = modelMatrix * endPoint;
                Vec3f viewEnd = viewMatrix * worldEnd;
                Vec3f clipEnd = projectionMatrix * viewEnd;
                Vec3f screenEnd = vpMat * clipEnd;

                bool inScreen = screenEnd.x >= 0 && screenEnd.x < screenW && 
                              screenEnd.y >= 0 && screenEnd.y < screenH;

                if (inScreen) {
                    if (!foundStart) {
                        // 如果这是第一个可见点，且之前有不可见点，绘制到这个点
                        if (currentLength > 0) {
                            startPoint = lastPoint;
                            if (m_enableSSAA) {
                                drawLineHighRes(startPoint, endPoint, gridColor, 1.0f * m_ssaaScale);
                            } else {
                                drawLine(startPoint, endPoint, gridColor, 1.0f);
                            }
                        }
                        foundStart = true;
                    }
                    lastValidPoint = endPoint;
                    wasInScreen = true;
                }
                else if (wasInScreen && !hasDrawn) {
                    // 找到结束点，绘制这段网格线
                    if (m_enableSSAA) {
                        drawLineHighRes(startPoint, endPoint, gridColor, 1.0f * m_ssaaScale);
                    } else {
                        drawLine(startPoint, endPoint, gridColor, 1.0f);
                    }
                    hasDrawn = true;
                    break;
                }

                if (currentLength > 1000.0f) {
                    if (wasInScreen && !hasDrawn) {
                        if (m_enableSSAA) {
                            drawLineHighRes(startPoint, endPoint, gridColor, 1.0f * m_ssaaScale);
                        } else {
                            drawLine(startPoint, endPoint, gridColor, 1.0f);
                        }
                    }
                    break;
                }

                lastPoint = endPoint;
                currentLength += 1.0f;
            }
        }
    }
}

// 绘制光源位置（保持向后兼容，绘制第一个光源）
void Renderer::drawLightPosition() {
    if (!lights.empty()) {
        drawSingleLightPosition(lights[0]);
    }
}

// 绘制所有光源位置
void Renderer::drawAllLightPositions() {
    for (size_t i = 0; i < lights.size(); i++) {
        drawSingleLightPosition(lights[i], i);
    }
}

// 绘制单个光源位置
void Renderer::drawSingleLightPosition(const Light& light, int lightIndex) {
    // 将光源位置从本地坐标系变换到世界坐标系
    Vec3f worldLightPos = modelMatrix * light.position;
    
    // 将世界坐标的光源位置转换到屏幕坐标
    Vec3f viewLightPos = viewMatrix * worldLightPos;
    Vec3f clipLightPos = projectionMatrix * viewLightPos;
    Vec3f screenLightPos = viewportMatrix * clipLightPos;
    
    int lx = static_cast<int>(screenLightPos.x);
    int ly = static_cast<int>(screenLightPos.y);
    float lz = screenLightPos.z;
    
    // 根据光源颜色绘制光源标记
    Color lightColor = Color::fromVec3f(light.color * 255.0f);
    int radius = 5; // 半径为5像素
    
    for (int dx = -radius; dx <= radius; dx++) {
        for (int dy = -radius; dy <= radius; dy++) {
            float distance = std::sqrt(static_cast<float>(dx * dx + dy * dy));
            if (distance <= radius) {
                int px = lx + dx;
                int py = ly + dy;
                if (px >= 0 && px < this->width && py >= 0 && py < this->height) {
                    if (depthTest(px, py, lz)) {
                        // 根据距离中心的远近调整亮度，创造渐变效果
                        float alpha = 1.0f - (distance / radius);
                        Color pixelColor(
                            static_cast<unsigned char>(lightColor.r * alpha),
                            static_cast<unsigned char>(lightColor.g * alpha),
                            static_cast<unsigned char>(lightColor.b * alpha)
                        );
                        setPixel(px, py, pixelColor, lz);
                    }
                }
            }
        }
    }
    
    // 在光源周围绘制一个小的十字标记，便于识别
    for (int i = -8; i <= 8; i++) {
        // 水平线
        int px = lx + i;
        int py = ly;
        if (px >= 0 && px < this->width && py >= 0 && py < this->height) {
            if (depthTest(px, py, lz)) {
                setPixel(px, py, lightColor, lz);
            }
        }
        
        // 垂直线
        px = lx;
        py = ly + i;
        if (px >= 0 && px < this->width && py >= 0 && py < this->height) {
            if (depthTest(px, py, lz)) {
                setPixel(px, py, lightColor, lz);
            }
        }
    }
}

// 绘制光线到模型顶点
void Renderer::drawLightRays(const Model& model) {
    const auto& vertices = model.getVertices();
    
    // 为每个光源使用不同的颜色
    std::vector<Color> rayColors = {
        Color(255, 255, 150, 100),  // 浅黄色 - 第一个光源
        Color(255, 150, 255, 100)   // 浅紫色 - 第二个光源
    };
    
    // 遍历所有顶点，绘制从每个光源到顶点的连线
    std::vector<Vec3f> uniquePositions;
    
    // 收集唯一的顶点位置（避免重复绘制到同一个位置的光线）
    for (const auto& vertex : vertices) {
        // 使用模型本地坐标的顶点位置
        Vec3f localVertexPos = vertex.position;
        
        // 检查是否已经有相同位置的顶点
        bool isDuplicate = false;
        for (const auto& pos : uniquePositions) {
            Vec3f diff = localVertexPos - pos;
            if (diff.length() < 0.01f) { // 如果距离小于0.01，认为是重复顶点
                isDuplicate = true;
                break;
            }
        }
        
        if (!isDuplicate) {
            uniquePositions.push_back(localVertexPos);
        }
    }
    
    // 为每个光源绘制光线
    for (size_t lightIndex = 0; lightIndex < lights.size() && lightIndex < rayColors.size(); lightIndex++) {
        const Light& light = lights[lightIndex];
        Color rayColor = rayColors[lightIndex];
        
        // 绘制从当前光源到每个唯一顶点位置的光线
        for (const auto& vertexPos : uniquePositions) {
            // 计算光线方向
            Vec3f lightToVertex = vertexPos - light.position;
            float distance = lightToVertex.length();
            
            // 只绘制一定距离内的光线，避免过长的线条
            if (distance < 10.0f) {
                // 在本地坐标系中绘制光线，drawLine会自动应用modelMatrix变换
                drawLine(light.position, vertexPos, rayColor, 1.0f);
            }
        }
    }
}

void Renderer::drawTriangleEdges(const ShaderVertex& v0, const ShaderVertex& v1, const ShaderVertex& v2) {
    // 绘制三角形边线 - 使用本地坐标，增加线宽防止断线
    drawLine(v0.localPos, v1.localPos, Color(0, 0, 0), 2.0f);  // 黑色边线，宽度2像素
    drawLine(v1.localPos, v2.localPos, Color(0, 0, 0), 2.0f);
    drawLine(v2.localPos, v0.localPos, Color(0, 0, 0), 2.0f);
}

void Renderer::drawTriangleEdgesHighRes(const ShaderVertex& v0, const ShaderVertex& v1, const ShaderVertex& v2) {
    // 在SSAA模式下绘制三角形边线到高分辨率缓冲区
    drawLineHighRes(v0.localPos, v1.localPos, Color(0, 0, 0), 2.0f * m_ssaaScale);  // 按比例调整线宽
    drawLineHighRes(v1.localPos, v2.localPos, Color(0, 0, 0), 2.0f * m_ssaaScale);
    drawLineHighRes(v2.localPos, v0.localPos, Color(0, 0, 0), 2.0f * m_ssaaScale);
}

// 新增：光源管理方法实现
void Renderer::setLight(int index, const Light& light) {
    if (index >= 0 && index < lights.size()) {
        lights[index] = light;
    }
}

void Renderer::removeLight(int index) {
    if (index >= 0 && index < lights.size()) {
        lights.erase(lights.begin() + index);
    }
}

const Light& Renderer::getLight(int index) const {
    static Light defaultLight;
    if (index >= 0 && index < lights.size()) {
        return lights[index];
    }
    return defaultLight;
}

Light& Renderer::getLight(int index) {
    static Light defaultLight;
    if (index >= 0 && index < lights.size()) {
        return lights[index];
    }
    return defaultLight;
}

Vec3f Renderer::calculateBRDFLighting(const Vec3f& localPos, const Vec3f& localNormal, const Vec3f& baseColor) {
    // 环境光
    float ambientOcclusion = (1.0f - m_metallic) * (1.0f - m_roughness);
    Vec3f ambient = baseColor * (ambientIntensity * ambientStrength) * ambientOcclusion;
    
    // 初始化总光照贡献
    Vec3f totalLighting = ambient;
    
    // 计算所有光源的 BRDF 贡献
    for (const auto& light : lights) {
        if (light.enabled) {
            totalLighting += calculateSingleLightBRDF(light, localPos, localNormal, baseColor);
        }
    }
    
    return totalLighting;
}

Vec3f Renderer::calculateSingleLightBRDF(const Light& light, const Vec3f& localPos, const Vec3f& localNormal, const Vec3f& baseColor) {
    Vec3f L;  // 光照方向
    float distance;
    
    if (light.type == LightType::DIRECTIONAL) {
        // 平面光源：使用固定方向
        L = -light.position.normalize();  // 光源position作为方向向量，取反得到光照方向
        distance = 1.0f;  // 平面光源无距离概念，设为1避免除零
    } else {
        // 点光源：原有逻辑
        Vec3f lightVector = light.position - localPos;
        distance = lightVector.length();
        L = lightVector.normalize();  // 光照方向
    }
    Vec3f N = localNormal.normalize();  // 法线方向
    
    // 计算视角方向
    Vec3f worldCameraPos = VectorMath::inverse(viewMatrix) * Vec3f(0, 0, 0);
    Matrix4x4 invModelMatrix = VectorMath::inverse(modelMatrix);
    Vec3f localCameraPos = invModelMatrix * worldCameraPos;
    Vec3f V = (localCameraPos - localPos).normalize();  // 视角方向
    
    // 半程向量
    Vec3f H = (V + L).normalize();
    
    // 计算各种点积
    float NdotV = std::max(0.00001f, N.dot(V));
    float NdotL = std::max(0.00001f, N.dot(L));
    float HdotV = std::max(0.0f, H.dot(V));
    
    // 如果表面背向光源，返回黑色
    if (NdotL <= 0.0f) {
        return Vec3f(0, 0, 0);
    }
    
    // 距离衰减
    float attenuation;
    if (light.type == LightType::DIRECTIONAL) {
        attenuation = light.intensity;  // 平面光源无距离衰减
    } else {
        attenuation = light.intensity / (distance * distance);  // 点光源距离衰减
    }
    
    // Cook-Torrance BRDF 计算
    float D = distributionGGX(N, H, m_roughness, NdotV);           // 法线分布函数
    float G = geometrySmith(N, V, L, m_roughness);          // 几何遮蔽函数
    Vec3f F = fresnelSchlick(HdotV, m_F0);                  // 菲涅尔项
    
    // 计算镜面反射项
    Vec3f numerator = Vec3f(D, D, D) * G * F;
    float denominator = std::max(0.00001f, 4.0f * NdotV * NdotL); // 增加一个更大的epsilon来防止除零

    Vec3f specular = numerator / denominator;
    
    // 能量守恒：漫反射 = (1 - 镜面反射) * (1 - 金属度)
    Vec3f kS = F;  // 镜面反射系数
    Vec3f kD = Vec3f(1.0f, 1.0f, 1.0f) - kS;  // 漫反射系数
    kD = kD * (1.0f - m_metallic);  // 金属材质没有漫反射
    
    // Lambert 漫反射
    Vec3f diffuse = kD * baseColor * (1.0f / 3.14159265359f);
    
    // 新增：能量补偿计算
    Vec3f energyCompensation(0, 0, 0);
    if (m_enableEnergyCompensation && m_roughness > 0.01f) {
        energyCompensation = calculateEnergyCompensationTerm(N, V, L, m_roughness, m_F0);
        energyCompensation = energyCompensation * m_energyCompensationScale;
    }
    
    // 最终颜色
    Vec3f radiance = light.color * attenuation;
    Vec3f color = (diffuse + specular + energyCompensation) * radiance * NdotL;
    
    // BRDF 不使用 Blinn-Phong 的 diffuseStrength 系数
    // BRDF 本身已经包含正确的能量分配，不需要额外缩放
    
    return color;
}

// GGX 法线分布函数 (Trowbridge-Reitz)
float Renderer::distributionGGX(const Vec3f& N, const Vec3f& H, float roughness, float NdotV) const {
    // 在掠射角时增加有效粗糙度
    float view_dependent_roughness = roughness;
    
    // 当视角接近掠射角时(NdotV接近0)，增加粗糙度
    if (NdotV < 0.5f && roughness <= 0.2f) {
        float t = 1.0f - (NdotV * 2.0f);  // 从0到1的过渡
        //view_dependent_roughness = std::min(1.0f, roughness + t * roughness);  // 最多增加0.5的粗糙度
    }
    
    float a = view_dependent_roughness * view_dependent_roughness;
    float a2 = a * a;
    float NdotH = std::max(0.0f, N.dot(H));
    float NdotH2 = NdotH * NdotH;
    
    float num = a2;
    float denom = (NdotH2 * (a2 - 1.0f) + 1.0f);
    denom = M_PI * denom * denom;
    
    return num / denom;
}

// Smith 几何遮蔽函数
float Renderer::geometrySchlickGGX(float NdotV, float roughness) const {
    float r = (roughness + 1.0f);
    float k = (r * r) / 8.0f;
    
    float num = NdotV;
    float denom = NdotV * (1.0f - k) + k;
    
    return num / denom;
}

float Renderer::geometrySmith(const Vec3f& N, const Vec3f& V, const Vec3f& L, float roughness) const {
    float NdotV = std::max(0.00001f, N.dot(V)); 
    float NdotL = std::max(0.00001f, N.dot(L)); 
    float ggx2 = geometrySchlickGGX(NdotV, roughness);
    float ggx1 = geometrySchlickGGX(NdotL, roughness);
    return ggx1 * ggx2;
}

// Fresnel 方程 (Schlick 近似)
Vec3f Renderer::fresnelSchlick(float cosTheta, const Vec3f& F0) const {
    float power = std::pow(1.0f - cosTheta, 5.0f);
    return F0 + (Vec3f(1.0f, 1.0f, 1.0f) - F0) * power;
}

// 新增：初始化能量补偿查找表
void Renderer::initializeEnergyCompensationLUT() const {
    if (m_lutInitialized) return;
    
    // 初始化查找表尺寸
    m_energyCompensationLUT.resize(LUT_SIZE);
    for (int i = 0; i < LUT_SIZE; i++) {
        m_energyCompensationLUT[i].resize(LUT_SIZE);
    }
    
    // 预计算能量补偿值
    for (int roughnessIdx = 0; roughnessIdx < LUT_SIZE; roughnessIdx++) {
        for (int cosThetaIdx = 0; cosThetaIdx < LUT_SIZE; cosThetaIdx++) {
            float roughness = static_cast<float>(roughnessIdx) / (LUT_SIZE - 1);
            float cosTheta = static_cast<float>(cosThetaIdx) / (LUT_SIZE - 1);
            
            // 防止除零
            roughness = std::max(0.0f, roughness);
            cosTheta = std::max(0.0f, cosTheta);
            
            // 计算能量积分
            m_energyCompensationLUT[roughnessIdx][cosThetaIdx] = computeEnergyIntegral(roughness, cosTheta);
        }
    }
    
    m_lutInitialized = true;
}

// 新增：查找能量补偿值
float Renderer::lookupEnergyCompensation(float roughness, float cosTheta) const {
    // 确保查找表已初始化
    initializeEnergyCompensationLUT();
    
    // 将输入值映射到查找表索引
    roughness = std::max(0.0001f, std::min(1.0f, roughness));
    cosTheta = std::max(0.0001f, std::min(1.0f, cosTheta));
    
    float roughnessFloat = roughness * (LUT_SIZE - 1);
    float cosThetaFloat = cosTheta * (LUT_SIZE - 1);
    
    int roughnessIdx0 = static_cast<int>(roughnessFloat);
    int cosThetaIdx0 = static_cast<int>(cosThetaFloat);
    int roughnessIdx1 = std::min(roughnessIdx0 + 1, LUT_SIZE - 1);
    int cosThetaIdx1 = std::min(cosThetaIdx0 + 1, LUT_SIZE - 1);
    
    float roughnessFrac = roughnessFloat - roughnessIdx0;
    float cosThetaFrac = cosThetaFloat - cosThetaIdx0;
    
    // 双线性插值
    float v00 = m_energyCompensationLUT[roughnessIdx0][cosThetaIdx0];
    float v01 = m_energyCompensationLUT[roughnessIdx0][cosThetaIdx1];
    float v10 = m_energyCompensationLUT[roughnessIdx1][cosThetaIdx0];
    float v11 = m_energyCompensationLUT[roughnessIdx1][cosThetaIdx1];
    
    float v0 = v00 * (1.0f - cosThetaFrac) + v01 * cosThetaFrac;
    float v1 = v10 * (1.0f - cosThetaFrac) + v11 * cosThetaFrac;
    
    return v0 * (1.0f - roughnessFrac) + v1 * roughnessFrac;
}

// 新增：计算能量积分（简化的蒙特卡洛积分）
float Renderer::computeEnergyIntegral(float roughness, float cosTheta) const {
    const int numSamples = 32;  // 样本数量
    float totalEnergy = 0.0f;
    
    // 简化的蒙特卡洛积分
    for (int i = 0; i < numSamples; i++) {
        for (int j = 0; j < numSamples; j++) {
            // 生成半球面采样方向
            float xi1 = static_cast<float>(i) / numSamples;
            float xi2 = static_cast<float>(j) / numSamples;
            
            // 重要性采样：基于GGX分布
            float a = roughness * roughness;
            float a2 = a * a;
            
            float theta = std::acos(std::sqrt((1.0f - xi1) / (xi1 * (a2 - 1.0f) + 1.0f)));
            float phi = 2.0f * M_PI * xi2;
            
            // 计算采样方向
            float sinTheta = std::sin(theta);
            Vec3f sampleDir(sinTheta * std::cos(phi), sinTheta * std::sin(phi), std::cos(theta));
            
            // 假设视角方向为(0,0,1)
            Vec3f V(0, 0, 1);
            Vec3f N(0, 0, 1);
            Vec3f L = sampleDir;
            Vec3f H = (V + L).normalize();
            
            float NdotL = std::max(0.0001f, N.dot(L));
            float NdotV = std::max(0.0001f, N.dot(V));
            float VdotH = std::max(0.0f, V.dot(H));
            
            if (NdotL > 0.0f && NdotV > 0.0f) {
                // 计算BRDF项
                float D = distributionGGX(N, H, roughness, NdotV);
                float G = geometrySmith(N, V, L, roughness);
                
                // 计算能量贡献
                float brdfContrib = (D * G) / (4.0f * NdotV * NdotL);
                totalEnergy += brdfContrib * NdotL;
            }
        }
    }
    
    // 归一化
    totalEnergy /= (numSamples * numSamples);
    
    // 计算能量损失：理想情况下应该为1.0，实际小于1.0的部分就是损失
    float energyLoss = std::max(0.0f, 1.0f - totalEnergy);
    
    return energyLoss;
}

// 新增：计算能量补偿项
Vec3f Renderer::calculateEnergyCompensationTerm(const Vec3f& N, const Vec3f& V, const Vec3f& L, 
                                                 float roughness, const Vec3f& F0) const {
    float NdotV = std::max(0.0001f, N.dot(V));
    float NdotL = std::max(0.0001f, N.dot(L));
    
    // 简化能量补偿计算
    float energyLoss = 1.0f - roughness;  // 粗糙度越高，能量损失越小
    
    // 计算菲涅尔项
    Vec3f H = (V + L).normalize();
    float HdotV = std::max(0.0f, H.dot(V));
    Vec3f F = fresnelSchlick(HdotV, F0);
    
    // 简化的能量补偿：基于粗糙度和菲涅尔项
    Vec3f energyCompensation = F * energyLoss * 0.5f;
    
    return energyCompensation;
}

// 新增：计算海胆刺位移
Vec3f Renderer::calculateSeaUrchinDisplacement(const Vec3f& position, const Vec3f& normal) const {
    if (!m_enableDisplacement) {
        return Vec3f(0, 0, 0);
    }
    
    // 使用球面坐标作为基础
    float theta = std::atan2(position.y, position.x);
    float phi = std::acos(position.z / position.length());
    
    // 生成噪声模式
    float noise = std::sin(theta * m_displacementFrequency) * 
                 std::cos(phi * m_displacementFrequency);
    
    // 添加更多细节
    noise += 0.5f * std::sin(theta * m_displacementFrequency * 2.0f) * 
             std::cos(phi * m_displacementFrequency * 2.0f);
    
    // 应用锐利度
    noise = std::pow(std::abs(noise), m_spineSharpness) * (noise >= 0 ? 1 : -1);
    
    // 计算位移量
    float displacement = noise * m_displacementScale * m_spineLength;
    
    // 沿法线方向位移
    return normal * displacement;
}

// 片段着色器结构体（本文件内部扩展）
struct LocalShaderFragment {
    Vec3f position;
    Vec3f normal;
    Vec2f texCoord;
    Vec3f worldPos;
    Vec3f localPos;
    Vec3f localNormal;
    int faceIndex = -1;
    float alpha = 1.0f;
};

// 新增：带face index的渲染流程
void Renderer::rasterizeTriangleWithFaceIdx(const ShaderVertex& v0, const ShaderVertex& v1, const ShaderVertex& v2, int faceIdx, const Model* pModel) {
    float x0 = v0.position.x;
    float y0 = v0.position.y;
    float x1 = v1.position.x;
    float y1 = v1.position.y;
    float x2 = v2.position.x;
    float y2 = v2.position.y;
    int minX = std::max(0, static_cast<int>(std::floor(std::min({x0, x1, x2}))));
    int maxX = std::min(width - 1, static_cast<int>(std::ceil(std::max({x0, x1, x2}))));
    int minY = std::max(0, static_cast<int>(std::floor(std::min({y0, y1, y2}))));
    int maxY = std::min(height - 1, static_cast<int>(std::ceil(std::max({y0, y1, y2}))));
    float faceAlpha = 1.0f;
    if (pModel && faceIdx >= 0) faceAlpha = pModel->getFaceAlpha(faceIdx);
    for (int y = minY; y <= maxY; y++) {
        for (int x = minX; x <= maxX; x++) {
            Vec3f bary = barycentric(Vec2f(x0, y0), Vec2f(x1, y1), Vec2f(x2, y2), Vec2f(static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f));
            if (bary.x >= 0 && bary.y >= 0 && bary.z >= 0) {
                float depth = bary.x * v0.position.z + bary.y * v1.position.z + bary.z * v2.position.z;
                if (depthTest(x, y, depth)) {
                    LocalShaderFragment fragment;
                    fragment.position = v0.position * bary.x + v1.position * bary.y + v2.position * bary.z;
                    fragment.normal = v0.normal * bary.x + v1.normal * bary.y + v2.normal * bary.z;
                    fragment.texCoord = v0.texCoord * bary.x + v1.texCoord * bary.y + v2.texCoord * bary.z;
                    fragment.worldPos = v0.worldPos * bary.x + v1.worldPos * bary.y + v2.worldPos * bary.z;
                    fragment.localPos = v0.localPos * bary.x + v1.localPos * bary.y + v2.localPos * bary.z;
                    fragment.localNormal = v0.localNormal * bary.x + v1.localNormal * bary.y + v2.localNormal * bary.z;
                    fragment.normal = fragment.normal.normalize();
                    fragment.localNormal = fragment.localNormal.normalize();
                    fragment.faceIndex = faceIdx;
                    fragment.alpha = faceAlpha;
                    Color color = fragmentShaderWithFaceIdx(fragment, pModel);
                    setPixelAlpha(x, y, color, depth, fragment.alpha);
                }
            }
        }
    }
}

// 新增：带face index的fragmentShader
Color Renderer::fragmentShaderWithFaceIdx(const LocalShaderFragment& fragment, const Model* pModel) {
    Vec3f baseColor(1, 1, 1);
    if (m_enableTexture && currentTexture && currentTexture->isValid()) {
        baseColor = currentTexture->sampleVec3f(fragment.texCoord.x, fragment.texCoord.y);
    }
    Vec3f normal = fragment.localNormal;
    if (m_enableNormalMap && currentNormalMap && currentNormalMap->isValid()) {
        Vec3f normalMapSample = currentNormalMap->sampleVec3f(fragment.texCoord.x, fragment.texCoord.y);
        Vec3f tangentSpaceNormal = normalMapSample * 2.0f - Vec3f(1.0f, 1.0f, 1.0f);
        Vec3f N = fragment.localNormal.normalize();
        Vec3f up = (std::abs(N.y) < 0.9f) ? Vec3f(0, 1, 0) : Vec3f(1, 0, 0);
        Vec3f T = up.cross(N).normalize();
        Vec3f B = N.cross(T);
        normal = T * tangentSpaceNormal.x + B * tangentSpaceNormal.y + N * tangentSpaceNormal.z;
        normal = normal.normalize();
    }
    Vec3f finalColor;
    if (!m_enableNormalMap && !m_enableBRDF) {
        // 仅Phong模型下，优先用缓存参数
        Vec3f ka = Vec3f(-1, -1, -1);
        Vec3f kd(-1,-1,-1), ks(-1,-1,-1), ke(-1,-1,-1);
        if (pModel && fragment.faceIndex >= 0) {
            ka = pModel->getFaceKa(fragment.faceIndex);
            kd = pModel->getFaceKd(fragment.faceIndex);
            ks = pModel->getFaceKs(fragment.faceIndex);
            ke = pModel->getFaceKe(fragment.faceIndex);
        }
        Vec3f useKa = (ka.x >= 0 && ka.y >= 0 && ka.z >= 0) ? ka : ambientStrength;
        Vec3f useKd = (kd.x >= 0 && kd.y >= 0 && kd.z >= 0) ? kd : diffuseStrength;
        Vec3f useKs = (ks.x >= 0 && ks.y >= 0 && ks.z >= 0) ? ks : specularStrength;
        Vec3f useKe = (ke.x >= 0 && ke.y >= 0 && ke.z >= 0) ? ke : (m_enableEmission ? m_emissionColor * m_emissionStrength : Vec3f(0,0,0));
        finalColor = calculateLightingWithParams(fragment.localPos, normal, baseColor, useKa, useKd, useKs, useKe);
    } else if (!m_enableNormalMap && m_enableBRDF) {
        finalColor = calculateBRDFLighting(fragment.localPos, normal, baseColor);
    } else {
        finalColor = calculateLighting(fragment.localPos, normal, baseColor);
    }
    return Color::fromVec3f(finalColor);
}

// 新增：带参数的calculateLighting
Vec3f Renderer::calculateLightingWithParams(const Vec3f& localPos, const Vec3f& localNormal, const Vec3f& baseColor, const Vec3f& ka, const Vec3f& kd, const Vec3f& ks, const Vec3f& emissionColor) {
    Vec3f ambient = baseColor * (ambientIntensity * ka);
    Vec3f totalLighting = ambient;
    for (const auto& light : lights) {
        if (light.enabled) {
            totalLighting += calculateSingleLightWithParams(light, localPos, localNormal, baseColor, kd, ks);
        }
    }
    totalLighting += emissionColor;
    return totalLighting;
}

// 新增：带参数的单光源光照
Vec3f Renderer::calculateSingleLightWithParams(const Light& light, const Vec3f& localPos, const Vec3f& localNormal, const Vec3f& baseColor, const Vec3f& kd, const Vec3f& ks) {
    Vec3f lightDir;
    float attenuation;
    if (light.type == LightType::DIRECTIONAL) {
        lightDir = -light.position.normalize();
        attenuation = light.intensity;
    } else {
        Vec3f lightVector = light.position - localPos;
        float distance = lightVector.length();
        lightDir = lightVector.normalize();
        float r_squared = distance * distance;
        attenuation = light.intensity / r_squared;
    }
    float diffuseIntensity = std::max(0.0f, localNormal.dot(lightDir));
    Vec3f diffuse = baseColor * light.color * diffuseIntensity * attenuation;
    diffuse = Vec3f(diffuse.x * kd.x, diffuse.y * kd.y, diffuse.z * kd.z);
    Vec3f specular(0, 0, 0);
    if (diffuseIntensity > 0.0f) {
        Vec3f worldCameraPos = VectorMath::inverse(viewMatrix) * Vec3f(0, 0, 0);
        Matrix4x4 invModelMatrix = VectorMath::inverse(modelMatrix);
        Vec3f localCameraPos = invModelMatrix * worldCameraPos;
        Vec3f localViewDir = (localCameraPos - localPos).normalize();
        Vec3f reflectDir = localNormal * (2.0f * localNormal.dot(lightDir)) - lightDir;
        reflectDir = reflectDir.normalize();
        float specularIntensity = std::pow(std::max(0.0f, localViewDir.dot(reflectDir)), this->shininess);
        specular = light.color * specularIntensity * attenuation;
        specular = Vec3f(specular.x * ks.x, specular.y * ks.y, specular.z * ks.z);
    }
    return diffuse + specular;
}

// 新增：带alpha混合的setPixel
void Renderer::setPixelAlpha(int x, int y, const Color& src, float depth, float alpha) {
    if (x < 0 || x >= width || y < 0 || y >= height) return;
    int index = y * width + x;
    Color dst = frameBuffer[index].color;
    float a = std::max(0.0f, std::min(1.0f, alpha));
    Color out;
    out.r = static_cast<unsigned char>(src.r * a + dst.r * (1 - a));
    out.g = static_cast<unsigned char>(src.g * a + dst.g * (1 - a));
    out.b = static_cast<unsigned char>(src.b * a + dst.b * (1 - a));
    out.a = 255;
    frameBuffer[index].color = out;
    frameBuffer[index].depth = depth;
}

// ================= MSAA 栅格化实现 =================

void Renderer::rasterizeTriangleMSAA(const ShaderVertex& v0, const ShaderVertex& v1, const ShaderVertex& v2, int faceIdx, const Model* pModel) {
    // 保持浮点数精度
    float x0 = v0.position.x;
    float y0 = v0.position.y;
    float x1 = v1.position.x;
    float y1 = v1.position.y;
    float x2 = v2.position.x;
    float y2 = v2.position.y;
    
    // 计算边界框
    int minX = std::max(0, static_cast<int>(std::floor(std::min({x0, x1, x2}))));
    int maxX = std::min(width - 1, static_cast<int>(std::ceil(std::max({x0, x1, x2}))));
    int minY = std::max(0, static_cast<int>(std::floor(std::min({y0, y1, y2}))));
    int maxY = std::min(height - 1, static_cast<int>(std::ceil(std::max({y0, y1, y2}))));
    
    for (int y = minY; y <= maxY; y++) {
        for (int x = minX; x <= maxX; x++) {
            int pixelIndex = y * width + x;
            
            // 对每个采样点进行测试
            for (int sampleId = 0; sampleId < m_msaaSampleCount; ++sampleId) {
                // 计算采样点在像素内的位置
                Vec2f sampleOffset = m_samplePattern[sampleId];
                Vec2f samplePos(
                    static_cast<float>(x) + 0.5f + sampleOffset.x,
                    static_cast<float>(y) + 0.5f + sampleOffset.y
                );
                
                // 判断采样点是否在三角形内
                if (isPointInTriangle(samplePos, Vec2f(x0, y0), Vec2f(x1, y1), Vec2f(x2, y2))) {
                    // 计算重心坐标
                    Vec3f bary = barycentric(Vec2f(x0, y0), Vec2f(x1, y1), Vec2f(x2, y2), samplePos);
                    
                    if (bary.x >= 0 && bary.y >= 0 && bary.z >= 0) {
                        // 插值深度
                        float sampleDepth = bary.x * v0.position.z + bary.y * v1.position.z + bary.z * v2.position.z;
                        
                        // 深度测试
                        MSAASample& sample = m_msaaFrameBuffer[pixelIndex].samples[sampleId];
                        if (sampleDepth < sample.depth) {
                            sample.depth = sampleDepth;
                            sample.covered = true;
                            
                            // 插值片段属性
                            LocalShaderFragment fragment;
                            fragment.position = v0.position * bary.x + v1.position * bary.y + v2.position * bary.z;
                            fragment.normal = v0.normal * bary.x + v1.normal * bary.y + v2.normal * bary.z;
                            fragment.texCoord = v0.texCoord * bary.x + v1.texCoord * bary.y + v2.texCoord * bary.z;
                            fragment.worldPos = v0.worldPos * bary.x + v1.worldPos * bary.y + v2.worldPos * bary.z;
                            fragment.localPos = v0.localPos * bary.x + v1.localPos * bary.y + v2.localPos * bary.z;
                            fragment.localNormal = v0.localNormal * bary.x + v1.localNormal * bary.y + v2.localNormal * bary.z;
                            
                            fragment.normal = fragment.normal.normalize();
                            fragment.localNormal = fragment.localNormal.normalize();
                            fragment.faceIndex = faceIdx;
                            
                            // 获取面的透明度
                            float faceAlpha = 1.0f;
                            if (pModel && faceIdx >= 0) {
                                faceAlpha = pModel->getFaceAlpha(faceIdx);
                            }
                            fragment.alpha = faceAlpha;
                            
                            // 执行片段着色器并获取源颜色
                            Color srcColor = fragmentShaderWithFaceIdx(fragment, pModel);
                
                            // Alpha 混合：与非 MSAA 路径一致
                            float a = std::max(0.0f, std::min(1.0f, fragment.alpha));
                
                            Color dstColor = sample.color; // 采样点当前颜色（背景或先前片段）
                            Color outColor;
                            outColor.r = static_cast<unsigned char>(srcColor.r * a + dstColor.r * (1.0f - a));
                            outColor.g = static_cast<unsigned char>(srcColor.g * a + dstColor.g * (1.0f - a));
                            outColor.b = static_cast<unsigned char>(srcColor.b * a + dstColor.b * (1.0f - a));
                            outColor.a = 255;
                
                            sample.color = outColor;
                        }
                    }
                }
            }
        }
    }
}

// ======== SSAA 相关新增实现 ========

// Alpha 混合写入高分辨率缓冲区
void Renderer::setPixelHighResAlpha(int x, int y, const Color& src, float depth, float alpha) {
    if (x < 0 || x >= m_highResWidth || y < 0 || y >= m_highResHeight) return;
    int index = y * m_highResWidth + x;
    Color dst = m_highResFrameBuffer[index].color;
    float a = std::max(0.0f, std::min(1.0f, alpha));
    Color out;
    out.r = static_cast<unsigned char>(src.r * a + dst.r * (1.0f - a));
    out.g = static_cast<unsigned char>(src.g * a + dst.g * (1.0f - a));
    out.b = static_cast<unsigned char>(src.b * a + dst.b * (1.0f - a));
    out.a = 255;
    m_highResFrameBuffer[index].color = out;
    m_highResFrameBuffer[index].depth = depth;
}

// 高分辨率栅格化（支持面索引/透明度）
void Renderer::rasterizeTriangleHighResWithFaceIdx(const ShaderVertex& v0, const ShaderVertex& v1, const ShaderVertex& v2, int faceIdx, const Model* pModel) {
    float x0 = v0.position.x, y0 = v0.position.y;
    float x1 = v1.position.x, y1 = v1.position.y;
    float x2 = v2.position.x, y2 = v2.position.y;

    int minX = std::max(0, static_cast<int>(std::floor(std::min({x0, x1, x2}))));
    int maxX = std::min(m_highResWidth - 1, static_cast<int>(std::ceil(std::max({x0, x1, x2}))));
    int minY = std::max(0, static_cast<int>(std::floor(std::min({y0, y1, y2}))));
    int maxY = std::min(m_highResHeight - 1, static_cast<int>(std::ceil(std::max({y0, y1, y2}))));

    for (int y = minY; y <= maxY; ++y) {
        for (int x = minX; x <= maxX; ++x) {
            Vec3f bary = barycentric(Vec2f(x0, y0), Vec2f(x1, y1), Vec2f(x2, y2),
                                     Vec2f(static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f));
            if (bary.x >= 0 && bary.y >= 0 && bary.z >= 0) {
                float depth = bary.x * v0.position.z + bary.y * v1.position.z + bary.z * v2.position.z;
                if (depthTestHighRes(x, y, depth)) {
                    LocalShaderFragment fragment;
                    fragment.position     = v0.position * bary.x + v1.position * bary.y + v2.position * bary.z;
                    fragment.normal       = v0.normal   * bary.x + v1.normal   * bary.y + v2.normal   * bary.z;
                    fragment.texCoord     = v0.texCoord * bary.x + v1.texCoord * bary.y + v2.texCoord * bary.z;
                    fragment.worldPos     = v0.worldPos * bary.x + v1.worldPos * bary.y + v2.worldPos * bary.z;
                    fragment.localPos     = v0.localPos * bary.x + v1.localPos * bary.y + v2.localPos * bary.z;
                    fragment.localNormal  = v0.localNormal * bary.x + v1.localNormal * bary.y + v2.localNormal * bary.z;
                    fragment.normal = fragment.normal.normalize();
                    fragment.localNormal = fragment.localNormal.normalize();
                    fragment.faceIndex = faceIdx;
                    float faceAlpha = 1.0f;
                    if (pModel && faceIdx >= 0) {
                        faceAlpha = pModel->getFaceAlpha(faceIdx);
                    }
                    fragment.alpha = faceAlpha;
                    Color srcColor = fragmentShaderWithFaceIdx(fragment, pModel);
                    setPixelHighResAlpha(x, y, srcColor, depth, fragment.alpha);
                }
            }
        }
    }
}

void Renderer::renderTriangleHighResWithFaceIdx(const Vertex& v0, const Vertex& v1, const Vertex& v2, int faceIdx, const Model* pModel) {
    ShaderVertex sv0 = vertexShader(v0);
    ShaderVertex sv1 = vertexShader(v1);
    ShaderVertex sv2 = vertexShader(v2);

    Vec3f worldPos0 = modelMatrix * v0.position;
    Vec3f worldPos1 = modelMatrix * v1.position;
    Vec3f worldPos2 = modelMatrix * v2.position;
    Vec3f viewPos0  = viewMatrix * worldPos0;
    Vec3f viewPos1  = viewMatrix * worldPos1;
    Vec3f viewPos2  = viewMatrix * worldPos2;
    if (!isFrontFace(viewPos0, viewPos1, viewPos2)) return;

    rasterizeTriangleHighResWithFaceIdx(sv0, sv1, sv2, faceIdx, pModel);
    if (m_drawTriangleEdges) {
        drawTriangleEdgesHighRes(sv0, sv1, sv2);
    }
}

// 将高分辨率缓冲区降采样到常规帧缓冲
void Renderer::resolveSSAAToFrameBuffer() {
    if (!m_enableSSAA) return;
#ifdef USE_CUDA
    downsampleFromHighResCUDA();
#else
    downsampleFromHighRes();
#endif
}

