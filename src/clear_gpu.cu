#include <cuda_runtime.h>

struct ColorDevice { unsigned char r, g, b, a; };
struct PixelDevice { ColorDevice color; float depth; };

__global__ void clearKernel(PixelDevice* buf, int count, ColorDevice col, float depth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    buf[idx].color = col;
    buf[idx].depth = depth;
}

extern "C" void cudaClearFrameBuffer(void* buf, int count,
                                      unsigned char r, unsigned char g,
                                      unsigned char b, unsigned char a,
                                      float depth) {
    PixelDevice* d_buf = nullptr;
    size_t bytes = sizeof(PixelDevice) * count;
    cudaMalloc(&d_buf, bytes);

    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    ColorDevice col = { r, g, b, a };
    clearKernel<<<blocks, threads>>>(d_buf, count, col, depth);
    cudaDeviceSynchronize();

    cudaMemcpy(buf, d_buf, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_buf);
} 