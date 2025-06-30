#include <cuda_runtime.h>
#include <math.h>

struct ColorDevice { unsigned char r, g, b, a; };
struct VertexDevice { float x, y, z; };
struct PixelDevice { ColorDevice color; float depth; };

__device__ inline bool barycentricInside(float u, float v, float w) {
    return u >= 0.0f && v >= 0.0f && w >= 0.0f;
}

__device__ float atomicMinFloat(float* addr, float value) {
    int* addr_i = reinterpret_cast<int*>(addr);
    int old_i = *addr_i, assumed;
    while (value < __int_as_float(old_i)) {
        assumed = old_i;
        old_i = atomicCAS(addr_i, assumed, __float_as_int(value));
        if (assumed == old_i) break;
    }
    return __int_as_float(old_i);
}

__global__ void rasterizeKernel(const VertexDevice* vertices, const int* indices,
                                int triCount, int width, int height,
                                PixelDevice* frameBuf) {
    int triId = blockIdx.x;
    if (triId >= triCount) return;

    // 三角形顶点索引
    int i0 = indices[triId * 3 + 0];
    int i1 = indices[triId * 3 + 1];
    int i2 = indices[triId * 3 + 2];
    VertexDevice v0 = vertices[i0];
    VertexDevice v1 = vertices[i1];
    VertexDevice v2 = vertices[i2];

    // 包围盒
    int minX = max(0, (int)floorf(fminf(fminf(v0.x, v1.x), v2.x)));
    int maxX = min(width - 1, (int)ceilf(fmaxf(fmaxf(v0.x, v1.x), v2.x)));
    int minY = max(0, (int)floorf(fminf(fminf(v0.y, v1.y), v2.y)));
    int maxY = min(height - 1, (int)ceilf(fmaxf(fmaxf(v0.y, v1.y), v2.y)));

    // 重心分母
    float denom = ((v1.y - v2.y)*(v0.x - v2.x) + (v2.x - v1.x)*(v0.y - v2.y));
    if (fabsf(denom) < 1e-6f) return;

    // 每线程扫描不同 y 行
    for (int py = minY + threadIdx.x; py <= maxY; py += blockDim.x) {
        for (int px = minX; px <= maxX; ++px) {
            float u = ((v1.y - v2.y)*(px - v2.x) + (v2.x - v1.x)*(py - v2.y)) / denom;
            float v = ((v2.y - v0.y)*(px - v2.x) + (v0.x - v2.x)*(py - v2.y)) / denom;
            float w = 1.0f - u - v;
            if (!barycentricInside(u, v, w)) continue;
            float depth = u * v0.z + v * v1.z + w * v2.z;
            int index = py * width + px;
            float prev = atomicMinFloat(&frameBuf[index].depth, depth);
            if (depth < prev) {
                frameBuf[index].color = { 20, 200, 20, 255 };
            }
        }
    }
}

extern "C" void launchRasterizeKernel(const void* d_vertices, const void* d_indices,
                                       int triCount, int width, int height,
                                       void* d_frameBuffer) {
    dim3 block(128);
    dim3 grid(triCount);
    rasterizeKernel<<<grid, block>>>((const VertexDevice*)d_vertices, (const int*)d_indices,
                                     triCount, width, height,
                                     (PixelDevice*)d_frameBuffer);
    cudaDeviceSynchronize();
} 