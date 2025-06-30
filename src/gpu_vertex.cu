#include <cuda_runtime.h>

struct VertexIn {
    float x, y, z;
};
struct VertexOut {
    float x, y, z;
};
__constant__ float d_mvp[16];

__device__ inline float4 mulMVP(float3 p) {
    float4 res;
    res.x = d_mvp[0]*p.x + d_mvp[4]*p.y + d_mvp[8]*p.z + d_mvp[12];
    res.y = d_mvp[1]*p.x + d_mvp[5]*p.y + d_mvp[9]*p.z + d_mvp[13];
    res.z = d_mvp[2]*p.x + d_mvp[6]*p.y + d_mvp[10]*p.z + d_mvp[14];
    res.w = d_mvp[3]*p.x + d_mvp[7]*p.y + d_mvp[11]*p.z + d_mvp[15];
    return res;
}

__global__ void vertexShaderKernel(const VertexIn* vin, VertexOut* vout, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    float3 pos = make_float3(vin[idx].x, vin[idx].y, vin[idx].z);
    float4 clip = mulMVP(pos);
    float invW = 1.0f / clip.w;
    vout[idx].x = clip.x * invW;
    vout[idx].y = clip.y * invW;
    vout[idx].z = clip.z * invW;
}

extern "C" void launchVertexShaderKernel(const void* verticesIn, void* verticesOut, int count, const float* mvp) {
    cudaMemcpyToSymbol(d_mvp, mvp, sizeof(float)*16);
    int threads=256;
    int blocks=(count+threads-1)/threads;
    vertexShaderKernel<<<blocks,threads>>>((const VertexIn*)verticesIn, (VertexOut*)verticesOut, count);
    cudaDeviceSynchronize();
} 