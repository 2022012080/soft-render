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
                                      float depth)
{
    if (buf == nullptr || count <= 0) return;

#if CUDART_VERSION >= 10000
    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, buf);
    bool isDevicePtr = (err == cudaSuccess && attr.type == cudaMemoryTypeDevice);
#else
    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, buf);
    bool isDevicePtr = (err == cudaSuccess && attr.memoryType == cudaMemoryTypeDevice);
#endif

    ColorDevice col = { r, g, b, a };

    if (isDevicePtr) {
        PixelDevice* d_buf = static_cast<PixelDevice*>(buf);
        int threads = 256;
        int blocks = (count + threads - 1) / threads;
        clearKernel<<<blocks, threads>>>(d_buf, count, col, depth);
        cudaDeviceSynchronize();
    } else {
        PixelDevice* h_buf = static_cast<PixelDevice*>(buf);
        for (int i = 0; i < count; ++i) {
            h_buf[i].color = col;
            h_buf[i].depth = depth;
        }
    }
} 