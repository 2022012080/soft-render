#include <cuda_runtime.h>

struct ColorDevice {
    unsigned char r, g, b, a;
};

struct PixelDevice {
    ColorDevice color;
    float depth;
};

__global__ void ssaaDownsampleKernel(const PixelDevice* hi, PixelDevice* lo,
                                     int lowW, int lowH, int scale, int highW)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= lowW || y >= lowH) return;

    int startX = x * scale;
    int startY = y * scale;
    int samples = scale * scale;

    int r = 0, g = 0, b = 0;
    float minDepth = 1.0f;

    for (int dy = 0; dy < scale; ++dy) {
        int rowIdx = (startY + dy) * highW + startX;
        for (int dx = 0; dx < scale; ++dx) {
            PixelDevice p = hi[rowIdx + dx];
            r += p.color.r;
            g += p.color.g;
            b += p.color.b;
            minDepth = fminf(minDepth, p.depth);
        }
    }

    PixelDevice out;
    out.color.r = static_cast<unsigned char>(r / samples);
    out.color.g = static_cast<unsigned char>(g / samples);
    out.color.b = static_cast<unsigned char>(b / samples);
    out.color.a = 255;
    out.depth = minDepth;

    lo[y * lowW + x] = out;
}

extern "C" void ssaaDownsampleKernelLauncher(const void* hi, void* lo,
                                              int lowW, int lowH, int scale, int highW)
{
    dim3 block(16, 16);
    dim3 grid((lowW + block.x - 1) / block.x, (lowH + block.y - 1) / block.y);
    ssaaDownsampleKernel<<<grid, block>>>((const PixelDevice*)hi, (PixelDevice*)lo,
                                         lowW, lowH, scale, highW);
    cudaDeviceSynchronize();
} 