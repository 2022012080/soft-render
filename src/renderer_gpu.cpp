#ifdef USE_CUDA
#include "renderer_gpu.h"
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <iostream>
#include <cstdint>

struct ColorDevice { unsigned char r, g, b, a; };
struct VertexDevice { float x, y, z; float nx, ny, nz; };
struct PixelDevice { ColorDevice color; float depth; };

struct LightDevice {
    int type;
    float pos[3];
    float color[3];
    float intensity;
    int enabled;
};
struct MaterialDevice {
    float ambient[3];
    float diffuse[3];
    float specular[3];
    float shininess;
};

extern "C" void launchRasterizeKernel(const void* d_vertices, const void* d_indices,
                                        int triCount, int width, int height,
                                        void* d_frameBuffer,
                                        const void* d_lights, int lightCount,
                                        const void* d_mat, float3 viewPos, float ambientIntensity,
                                        const void* d_faceAlphas,
                                        bool writeDepth);

extern "C" void cudaClearFrameBuffer(void* buf, int count,
                                      unsigned char r, unsigned char g,
                                      unsigned char b, unsigned char a,
                                      float depth);

extern "C" void launchVertexShaderKernel(const void*, void*, int, const float*, int, int);

#define CHECK_CUDA_ERROR(msg) { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        std::cout << "[CUDA ERROR] " << msg << ": " << cudaGetErrorString(err) << " (code " << err << ")" << std::endl; \
    } \
}

void RendererGPU::renderModel(const Model& model) {
    std::cout << "[GPU] renderModel called" << std::endl;
    int w = getWidth();
    int h = getHeight();

    std::cout << "[GPU] Preparing vertex/index data..." << std::endl;
    std::vector<VertexDevice> cpuVerts;
    std::vector<int> gpuIndicesOpaque;
    std::vector<int> gpuIndicesTransparent;
    std::vector<float> faceAlphasOpaque;
    std::vector<float> faceAlphasTrans;
    size_t faceCount = model.getFaceCount();
    cpuVerts.reserve(faceCount * 3);
    gpuIndicesOpaque.reserve(faceCount * 3);
    gpuIndicesTransparent.reserve(faceCount * 3);
    faceAlphasOpaque.reserve(faceCount);
    faceAlphasTrans.reserve(faceCount);

    // Device pointers
    VertexDevice* d_verts = nullptr;
    int* d_idx = nullptr;
    PixelDevice* d_frame = nullptr;
    VertexDevice* d_clip = nullptr;
    LightDevice* d_lights = nullptr;
    MaterialDevice* d_mat = nullptr;

    for (size_t i = 0; i < faceCount; ++i) {
        Vertex v0, v1, v2;
        model.getFaceVertices(static_cast<int>(i), v0, v1, v2);
        //Renderer::ShaderVertex sv0 = vertexShader(v0);
        //Renderer::ShaderVertex sv1 = vertexShader(v1);
        //Renderer::ShaderVertex sv2 = vertexShader(v2);

        auto addV = [&cpuVerts](const Vertex& v){
            VertexDevice vd{ v.position.x, v.position.y, v.position.z, v.normal.x, v.normal.y, v.normal.z };
            cpuVerts.push_back(vd);
        };
        addV(v0); addV(v1); addV(v2);
        int base = static_cast<int>(i * 3);
        float alpha = model.getFaceAlpha(static_cast<int>(i));
        if (alpha >= 0.999f) {
            gpuIndicesOpaque.push_back(base);
            gpuIndicesOpaque.push_back(base + 1);
            gpuIndicesOpaque.push_back(base + 2);
            faceAlphasOpaque.push_back(alpha);
        } else {
            gpuIndicesTransparent.push_back(base);
            gpuIndicesTransparent.push_back(base + 1);
            gpuIndicesTransparent.push_back(base + 2);
            faceAlphasTrans.push_back(alpha);
        }
    }

    size_t vBytes = cpuVerts.size() * sizeof(VertexDevice);
    size_t iBytesOpaque = gpuIndicesOpaque.size() * sizeof(int);
    size_t iBytesTrans = gpuIndicesTransparent.size() * sizeof(int);

    // For MVP
    float mvpArr[16];
    Matrix4x4 mvp = projectionMatrix * viewMatrix * modelMatrix;
    // 填充mvpArr为column-major顺序
    std::cout << "[DEBUG] MVP matrix (row major):\n";
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            std::cout << mvp.m[i][j] << " ";
        }
        std::cout << std::endl;
    }
    // 转置填充mvpArr（column-major）
    for (int j = 0, idx = 0; j < 4; ++j)
        for (int i = 0; i < 4; ++i, ++idx)
            mvpArr[idx] = mvp.m[i][j];
    std::cout << "[DEBUG] mvpArr (column-major): ";
    for (int i = 0; i < 16; ++i) std::cout << mvpArr[i] << " ";
    std::cout << std::endl;

    // For framebuffer clear
    Color black(0, 0, 0, 255);

    MaterialDevice mat;
    Vec3f ka(0.2f, 0.2f, 0.2f), kd(0.8f, 0.8f, 0.8f), ks(0.2f, 0.2f, 0.2f);
    float shininess = 32.0f;
    if (faceCount > 0) {
        ka = model.getFaceKa(0);
        kd = model.getFaceKd(0);
        ks = model.getFaceKs(0);
        // Use renderer's shininess
        shininess = getShininess();
    }
    mat.ambient[0] = ka.x; mat.ambient[1] = ka.y; mat.ambient[2] = ka.z;
    mat.diffuse[0] = kd.x; mat.diffuse[1] = kd.y; mat.diffuse[2] = kd.z;
    mat.specular[0] = ks.x; mat.specular[1] = ks.y; mat.specular[2] = ks.z;
    mat.shininess = shininess;

    std::vector<LightDevice> lightDevs;
    int lightCount = getLightCount();
    for (int i = 0; i < lightCount; ++i) {
        const Light& l = getLight(i);
        LightDevice ld;
        ld.type = (l.type == LightType::DIRECTIONAL) ? 1 : 0;
        ld.pos[0] = l.position.x; ld.pos[1] = l.position.y; ld.pos[2] = l.position.z;
        ld.color[0] = l.color.x; ld.color[1] = l.color.y; ld.color[2] = l.color.z;
        ld.intensity = l.intensity;
        ld.enabled = l.enabled ? 1 : 0;
        lightDevs.push_back(ld);
    }
    if (!lightDevs.empty()) {
        cudaMalloc(&d_lights, sizeof(LightDevice) * lightDevs.size());
        CHECK_CUDA_ERROR("cudaMalloc d_lights");
        cudaMemcpy(d_lights, lightDevs.data(), sizeof(LightDevice) * lightDevs.size(), cudaMemcpyHostToDevice);
        CHECK_CUDA_ERROR("cudaMemcpy d_lights");
    }

    cudaMalloc(&d_mat, sizeof(MaterialDevice));
    CHECK_CUDA_ERROR("cudaMalloc d_mat");
    cudaMemcpy(d_mat, &mat, sizeof(MaterialDevice), cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR("cudaMemcpy d_mat");

    Matrix4x4 invView = VectorMath::inverse(viewMatrix);
    Vec3f camWorld = invView * Vec3f(0, 0, 0);
    float3 viewPos = make_float3(camWorld.x, camWorld.y, camWorld.z);

    float ambientIntensity = getAmbientStrength().x; // Use x as representative

    std::cout << "[GPU] Allocating device memory for vertices..." << std::endl;
    cudaError_t err;
    err = cudaMalloc(&d_verts, vBytes); if (err) { std::cout << "[cudaMalloc][ERROR] d_verts failed: " << cudaGetErrorString(err) << std::endl; return; }
    CHECK_CUDA_ERROR("cudaMalloc d_verts");
    size_t maxIdxBytes = std::max(iBytesOpaque, iBytesTrans);
    err = cudaMalloc(&d_idx, std::max(maxIdxBytes, (size_t)1));
    if (err) { std::cout << "[cudaMalloc][ERROR] d_idx failed: " << cudaGetErrorString(err) << std::endl; return; }
    CHECK_CUDA_ERROR("cudaMalloc d_idx");
    std::cout << "[DEBUG] sizeof(PixelDevice) (host) = " << sizeof(PixelDevice) << std::endl;
    size_t frameBytes = w * h * sizeof(PixelDevice);
    std::cout << "[DEBUG] frameBytes = " << frameBytes << ", w = " << w << ", h = " << h << ", sizeof(PixelDevice) = " << sizeof(PixelDevice) << std::endl;
    err = cudaMalloc(&d_frame, frameBytes); if (err) { std::cout << "[cudaMalloc][ERROR] d_frame failed: " << cudaGetErrorString(err) << std::endl; return; }
    CHECK_CUDA_ERROR("cudaMalloc d_frame");
    // 清空framebuffer和depth
    cudaClearFrameBuffer(d_frame, w * h, black.r, black.g, black.b, black.a, 1.0f);
    CHECK_CUDA_ERROR("cudaClearFrameBuffer");

    std::cout << "[GPU] Copying vertex/index data to device..." << std::endl;
    err = cudaMemcpy(d_verts, cpuVerts.data(), vBytes, cudaMemcpyHostToDevice); if (err) { std::cout << "[cudaMemcpy][ERROR] d_verts failed: " << cudaGetErrorString(err) << std::endl; return; }
    CHECK_CUDA_ERROR("cudaMemcpy d_verts");
    if (!gpuIndicesOpaque.empty()) {
        err = cudaMemcpy(d_idx, gpuIndicesOpaque.data(), iBytesOpaque, cudaMemcpyHostToDevice);
        if (err) { std::cout << "[cudaMemcpy][ERROR] d_idx opaque failed: " << cudaGetErrorString(err) << std::endl; return; }
        CHECK_CUDA_ERROR("cudaMemcpy d_idx opaque");
    }

    std::cout << "[GPU] Allocating device memory for lights/material..." << std::endl;
    if (!lightDevs.empty()) {
        err = cudaMalloc(&d_lights, sizeof(LightDevice) * lightDevs.size()); if (err) { std::cout << "[cudaMalloc][ERROR] d_lights failed: " << cudaGetErrorString(err) << std::endl; return; }
        CHECK_CUDA_ERROR("cudaMalloc d_lights (2)");
        err = cudaMemcpy(d_lights, lightDevs.data(), sizeof(LightDevice) * lightDevs.size(), cudaMemcpyHostToDevice); if (err) { std::cout << "[cudaMemcpy][ERROR] d_lights failed: " << cudaGetErrorString(err) << std::endl; return; }
        CHECK_CUDA_ERROR("cudaMemcpy d_lights (2)");
    }
    err = cudaMalloc(&d_mat, sizeof(MaterialDevice)); if (err) { std::cout << "[cudaMalloc][ERROR] d_mat failed: " << cudaGetErrorString(err) << std::endl; return; }
    CHECK_CUDA_ERROR("cudaMalloc d_mat (2)");
    err = cudaMemcpy(d_mat, &mat, sizeof(MaterialDevice), cudaMemcpyHostToDevice); if (err) { std::cout << "[cudaMemcpy][ERROR] d_mat failed: " << cudaGetErrorString(err) << std::endl; return; }
    CHECK_CUDA_ERROR("cudaMemcpy d_mat (2)");

    std::cout << "[GPU] Allocating device memory for clip vertices..." << std::endl;
    err = cudaMalloc(&d_clip, vBytes); if (err) { std::cout << "[cudaMalloc][ERROR] d_clip failed: " << cudaGetErrorString(err) << std::endl; return; }
    CHECK_CUDA_ERROR("cudaMalloc d_clip");

    std::cout << "[DEBUG] cpuVerts.size() = " << cpuVerts.size() << std::endl;
    std::cout << "[DEBUG] vBytes = " << vBytes << std::endl;
    std::cout << "[DEBUG] d_verts = " << d_verts << std::endl;
    std::cout << "[DEBUG] d_clip = " << d_clip << std::endl;
    std::cout << "[DEBUG] mvpArr = ";
    for (int i = 0; i < 16; ++i) {
        std::cout << mvpArr[i] << " ";
    }
    std::cout << std::endl;

    // 启动kernel前，检查索引范围
    int maxIdx = 0;
    for (size_t i = 0; i < gpuIndicesOpaque.size(); ++i) {
        if (gpuIndicesOpaque[i] > maxIdx) maxIdx = gpuIndicesOpaque[i];
    }
    std::cout << "[DEBUG] max gpuIndices = " << maxIdx << ", cpuVerts.size() = " << cpuVerts.size() << std::endl;
    std::cout << "[DEBUG] gpuIndices.size() = " << gpuIndicesOpaque.size() << ", triCount = " << faceCount << std::endl;

    // 输出第一个三角形的原始顶点和MVP变换后坐标
    if (cpuVerts.size() >= 3) {
        std::cout << "[CPU][triId=0] v0=(" << cpuVerts[0].x << "," << cpuVerts[0].y << "," << cpuVerts[0].z << ") ";
        std::cout << "v1=(" << cpuVerts[1].x << "," << cpuVerts[1].y << "," << cpuVerts[1].z << ") ";
        std::cout << "v2=(" << cpuVerts[2].x << "," << cpuVerts[2].y << "," << cpuVerts[2].z << ")\n";
        // 经过MVP变换后的坐标
        for (int i = 0; i < 3; ++i) {
            float x = cpuVerts[i].x, y = cpuVerts[i].y, z = cpuVerts[i].z, w = 1.0f;
            float tx = mvp.m[0][0]*x + mvp.m[0][1]*y + mvp.m[0][2]*z + mvp.m[0][3]*w;
            float ty = mvp.m[1][0]*x + mvp.m[1][1]*y + mvp.m[1][2]*z + mvp.m[1][3]*w;
            float tz = mvp.m[2][0]*x + mvp.m[2][1]*y + mvp.m[2][2]*z + mvp.m[2][3]*w;
            float tw = mvp.m[3][0]*x + mvp.m[3][1]*y + mvp.m[3][2]*z + mvp.m[3][3]*w;
            std::cout << "[CPU][triId=0][MVP] v" << i << "_clip=(" << tx << "," << ty << "," << tz << "," << tw << ")\n";
        }
    }

    // Create device alpha arrays separately
    auto uploadAlphaArray = [&](const std::vector<float>& vec, float*& d_ptr){
        if (vec.empty()) { d_ptr = nullptr; return; }
        cudaMalloc(&d_ptr, sizeof(float)*vec.size());
        CHECK_CUDA_ERROR("cudaMalloc alpha");
        cudaMemcpy(d_ptr, vec.data(), sizeof(float)*vec.size(), cudaMemcpyHostToDevice);
        CHECK_CUDA_ERROR("cudaMemcpy alpha");
    };

    float* d_alphasOpaque = nullptr;
    float* d_alphasTrans  = nullptr;
    uploadAlphaArray(faceAlphasOpaque, d_alphasOpaque);
    uploadAlphaArray(faceAlphasTrans,  d_alphasTrans);

    launchVertexShaderKernel(d_verts, d_clip, static_cast<int>(cpuVerts.size()), mvpArr, w, h);
    CHECK_CUDA_ERROR("launchVertexShaderKernel");
    cudaFree(d_verts);
    d_verts = nullptr;

    // 调试：将d_clip拷贝回host，输出前三个clip顶点内容
    std::vector<VertexDevice> hostClipVerts(cpuVerts.size());
    cudaMemcpy(hostClipVerts.data(), d_clip, vBytes, cudaMemcpyDeviceToHost);
    std::cout << "[CPU][clip] triId=0 clip verts:" << std::endl;
    for (int i = 0; i < 3 && i < hostClipVerts.size(); ++i) {
        std::cout << "v" << i << "_clip=(" << hostClipVerts[i].x << "," << hostClipVerts[i].y << "," << hostClipVerts[i].z << ")\n";
    }

    // 统计CPU端第一个三角形实际写入像素数
    int cpuPixelWriteCount = 0;
    if (cpuVerts.size() >= 3) {
        // 取第一个三角形的clip顶点
        float x0 = hostClipVerts[0].x, y0 = hostClipVerts[0].y, z0 = hostClipVerts[0].z;
        float x1 = hostClipVerts[1].x, y1 = hostClipVerts[1].y, z1 = hostClipVerts[1].z;
        float x2 = hostClipVerts[2].x, y2 = hostClipVerts[2].y, z2 = hostClipVerts[2].z;
        int minX = std::max(0, static_cast<int>(std::floor(std::min({x0, x1, x2}))));
        int maxX = std::min(w - 1, static_cast<int>(std::ceil(std::max({x0, x1, x2}))));
        int minY = std::max(0, static_cast<int>(std::floor(std::min({y0, y1, y2}))));
        int maxY = std::min(h - 1, static_cast<int>(std::ceil(std::max({y0, y1, y2}))));
        float denom = (y1 - y2)*(x0 - x2) + (x2 - x1)*(y0 - y2);
        for (int y = minY; y <= maxY; ++y) {
            for (int x = minX; x <= maxX; ++x) {
                float u = ((y1 - y2)*((float)x + 0.5f - x2) + (x2 - x1)*((float)y + 0.5f - y2)) / denom;
                float v = ((y2 - y0)*((float)x + 0.5f - x2) + (x0 - x2)*((float)y + 0.5f - y2)) / denom;
                float w = 1.0f - u - v;
                if (u >= 0 && v >= 0 && w >= 0) {
                    cpuPixelWriteCount++;
                }
            }
        }
        std::cout << "[CPU][triId=0] pixelWriteCount = " << cpuPixelWriteCount << std::endl;
    }

    std::cout << "[GPU] Clearing framebuffer..." << std::endl;
    cudaClearFrameBuffer(getFrameBufferMutable().data(), w * h, black.r, black.g, black.b, black.a, 1.0f);
    CHECK_CUDA_ERROR("cudaClearFrameBuffer");
    err = cudaMemcpy(d_frame, getFrameBufferMutable().data(), frameBytes, cudaMemcpyHostToDevice); if (err) { std::cout << "[cudaMemcpy][ERROR] framebuffer failed: " << cudaGetErrorString(err) << std::endl; return; }
    CHECK_CUDA_ERROR("cudaMemcpy framebuffer");

    // Prepare timing events
    cudaEvent_t startKernel, stopKernel, startMemcpy, stopMemcpy;
    err = cudaEventCreate(&startKernel); if (err) std::cout << "cudaEventCreate startKernel error: " << err << std::endl;
    CHECK_CUDA_ERROR("cudaEventCreate startKernel");
    err = cudaEventCreate(&stopKernel); if (err) std::cout << "cudaEventCreate stopKernel error: " << err << std::endl;
    CHECK_CUDA_ERROR("cudaEventCreate stopKernel");
    err = cudaEventCreate(&startMemcpy); if (err) std::cout << "cudaEventCreate startMemcpy error: " << err << std::endl;
    CHECK_CUDA_ERROR("cudaEventCreate startMemcpy");
    err = cudaEventCreate(&stopMemcpy); if (err) std::cout << "cudaEventCreate stopMemcpy error: " << err << std::endl;
    CHECK_CUDA_ERROR("cudaEventCreate stopMemcpy");

    std::cout << "[GPU] Launching rasterizer kernel..." << std::endl;
    cudaEventRecord(startKernel, 0);
    CHECK_CUDA_ERROR("cudaEventRecord startKernel");
    int triCountOpaque = static_cast<int>(gpuIndicesOpaque.size() / 3);
    int triCountTrans  = static_cast<int>(gpuIndicesTransparent.size() / 3);

    // Pass 1: Opaque
    if (triCountOpaque > 0) {
        launchRasterizeKernel(d_clip, d_idx, triCountOpaque, w, h, d_frame,
            d_lights, lightCount, d_mat, viewPos, ambientIntensity, d_alphasOpaque, true);
        CHECK_CUDA_ERROR("launchRasterizeKernel opaque");
    }

    // Pass 2: Transparent --- sort back-to-front & render one triangle per launch
    if (triCountTrans > 0) {
        struct TransInfo { int baseIdx; float dist; float alpha; };
        std::vector<TransInfo> transList;
        transList.reserve(triCountTrans);

        // 计算相机世界坐标（已在 camWorld）
        for (size_t i = 0; i < triCountTrans; ++i) {
            int base = gpuIndicesTransparent[i*3];
            // 取三个顶点局部坐标 -> 世界坐标
            Vec3f lp0(cpuVerts[base].x, cpuVerts[base].y, cpuVerts[base].z);
            Vec3f lp1(cpuVerts[base+1].x, cpuVerts[base+1].y, cpuVerts[base+1].z);
            Vec3f lp2(cpuVerts[base+2].x, cpuVerts[base+2].y, cpuVerts[base+2].z);
            Vec3f wp0 = modelMatrix * lp0;
            Vec3f wp1 = modelMatrix * lp1;
            Vec3f wp2 = modelMatrix * lp2;
            Vec3f centroid = (wp0 + wp1 + wp2) / 3.0f;
            float dist = (centroid - camWorld).length();
            transList.push_back({base, dist, faceAlphasTrans[i]});
        }

        std::sort(transList.begin(), transList.end(), [](const TransInfo& a, const TransInfo& b){ return a.dist > b.dist; });

        // 设备端 alpha 缓冲（单值复用）
        float* d_alphaSingle = nullptr;
        cudaMalloc(&d_alphaSingle, sizeof(float));
        CHECK_CUDA_ERROR("cudaMalloc alphaSingle");

        int triplet[3];
        for (const auto& t : transList) {
            triplet[0] = t.baseIdx;
            triplet[1] = t.baseIdx + 1;
            triplet[2] = t.baseIdx + 2;
            cudaMemcpy(d_idx, triplet, sizeof(triplet), cudaMemcpyHostToDevice);
            cudaMemcpy(d_alphaSingle, &t.alpha, sizeof(float), cudaMemcpyHostToDevice);
            launchRasterizeKernel(d_clip, d_idx, 1, w, h, d_frame,
                d_lights, lightCount, d_mat, viewPos, ambientIntensity, d_alphaSingle, false);
            CHECK_CUDA_ERROR("launchRasterizeKernel trans tri");
        }

        cudaFree(d_alphaSingle);
    }

    cudaEventRecord(stopKernel, 0);
    CHECK_CUDA_ERROR("cudaEventRecord stopKernel");
    cudaEventSynchronize(stopKernel);
    CHECK_CUDA_ERROR("cudaEventSynchronize stopKernel");
    float kernelMs = 0.0f;
    cudaEventElapsedTime(&kernelMs, startKernel, stopKernel);
    CHECK_CUDA_ERROR("cudaEventElapsedTime kernel");
    std::cout << "[GPU] Rasterizer kernel finished." << std::endl;

    std::cout << "[GPU] Copying framebuffer from device to host..." << std::endl;
    cudaEventRecord(startMemcpy, 0);
    CHECK_CUDA_ERROR("cudaEventRecord startMemcpy");
    err = cudaMemcpy(getFrameBufferMutable().data(), d_frame, frameBytes, cudaMemcpyDeviceToHost); if (err) { std::cout << "[cudaMemcpy][ERROR] framebuffer (device to host) failed: " << cudaGetErrorString(err) << std::endl; return; }
    CHECK_CUDA_ERROR("cudaMemcpy framebuffer (device to host)");
    cudaEventRecord(stopMemcpy, 0);
    CHECK_CUDA_ERROR("cudaEventRecord stopMemcpy");
    cudaEventSynchronize(stopMemcpy);
    CHECK_CUDA_ERROR("cudaEventSynchronize stopMemcpy");
    float memcpyMs = 0.0f;
    cudaEventElapsedTime(&memcpyMs, startMemcpy, stopMemcpy);
    CHECK_CUDA_ERROR("cudaEventElapsedTime memcpy");
    std::cout << "[GPU] Framebuffer copy finished." << std::endl;

    std::cout << "[GPU] Kernel time: " << kernelMs << " ms, cudaMemcpy time: " << memcpyMs << " ms" << std::endl;

    cudaFree(d_idx);
    cudaFree(d_frame);
    if (d_lights) cudaFree(d_lights);
    if (d_mat) cudaFree(d_mat);
    cudaFree(d_clip);
    d_clip = nullptr;
    if (d_alphasOpaque) cudaFree(d_alphasOpaque);
    if (d_alphasTrans) cudaFree(d_alphasTrans);
    cudaEventDestroy(startKernel);
    cudaEventDestroy(stopKernel);
    cudaEventDestroy(startMemcpy);
    cudaEventDestroy(stopMemcpy);
    std::cout << "[GPU] renderModel finished." << std::endl;

    printf("[host] sizeof(PixelDevice)=%zu\n", sizeof(PixelDevice));
}

#endif // USE_CUDA 