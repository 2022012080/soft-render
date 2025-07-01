#include <cuda_runtime.h>

struct VertexIn {
    float x, y, z;
    float nx, ny, nz;
};
struct VertexOut {
    float x, y, z;
    float nx, ny, nz;
};
__constant__ float d_mvp[16];

__global__ void vertexShaderKernel(const VertexIn* vin, VertexOut* vout,
                                   int count, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    float3 inPos = make_float3(vin[idx].x, vin[idx].y, vin[idx].z);
    float4 clip;
    clip.x = d_mvp[0]*inPos.x + d_mvp[4]*inPos.y + d_mvp[8]*inPos.z + d_mvp[12];
    clip.y = d_mvp[1]*inPos.x + d_mvp[5]*inPos.y + d_mvp[9]*inPos.z + d_mvp[13];
    clip.z = d_mvp[2]*inPos.x + d_mvp[6]*inPos.y + d_mvp[10]*inPos.z + d_mvp[14];
    clip.w = d_mvp[3]*inPos.x + d_mvp[7]*inPos.y + d_mvp[11]*inPos.z + d_mvp[15];

    float invW = 1.0f / clip.w;
    float ndcX = clip.x * invW;
    float ndcY = clip.y * invW;
    float ndcZ = clip.z * invW; 
    float depth01 = (ndcZ + 1.0f) * 0.5f;  // 映射到[0,1]

    float screenX = (ndcX + 1.0f) * 0.5f * static_cast<float>(width);
    float screenY = (ndcY + 1.0f) * 0.5f * static_cast<float>(height);

    vout[idx].x = screenX;
    vout[idx].y = screenY;
    vout[idx].z = depth01; 
    vout[idx].nx = vin[idx].nx;
    vout[idx].ny = vin[idx].ny;
    vout[idx].nz = vin[idx].nz;
}

extern "C" void launchVertexShaderKernel(const void* verticesIn, void* verticesOut,
                                           int count, const float* mvp,
                                           int width, int height) {
    cudaMemcpyToSymbol(d_mvp, mvp, sizeof(float)*16);
    int threads = 256;
    int blocks  = (count + threads - 1) / threads;
    vertexShaderKernel<<<blocks, threads>>>((const VertexIn*)verticesIn,
                                            (VertexOut*)verticesOut,
                                            count, width, height);
    cudaDeviceSynchronize();
} 