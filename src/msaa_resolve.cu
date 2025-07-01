#include <cuda_runtime.h>

struct ColorDevice { unsigned char r, g, b, a; };
struct SampleDevice { ColorDevice color; float depth; unsigned char covered; };
struct PixelDevice { ColorDevice color; float depth; };

__device__ inline ColorDevice mixColor(const ColorDevice& c1, const ColorDevice& c2, float t) {
    ColorDevice out;
    out.r = static_cast<unsigned char>(c1.r * t + c2.r * (1.0f - t));
    out.g = static_cast<unsigned char>(c1.g * t + c2.g * (1.0f - t));
    out.b = static_cast<unsigned char>(c1.b * t + c2.b * (1.0f - t));
    out.a = 255;
    return out;
}

__global__ void msaaResolveKernel(const SampleDevice* samples, const ColorDevice* bg,
                                  PixelDevice* out, int pixelCount, int sampleCount)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pixelCount) return;

    const SampleDevice* pixelSamples = samples + idx * sampleCount;

    int coveredCount = 0;
    int sumR = 0, sumG = 0, sumB = 0;
    float minDepth = 1.0f;

    for (int s = 0; s < sampleCount; ++s) {
        if (pixelSamples[s].covered) {
            coveredCount++;
            sumR += pixelSamples[s].color.r;
            sumG += pixelSamples[s].color.g;
            sumB += pixelSamples[s].color.b;
            minDepth = fminf(minDepth, pixelSamples[s].depth);
        }
    }

    PixelDevice result;

    if (coveredCount == 0) {
        result.color = bg[idx];
        result.depth = 1.0f;
    } else {
        ColorDevice avg;
        avg.r = static_cast<unsigned char>(sumR / coveredCount);
        avg.g = static_cast<unsigned char>(sumG / coveredCount);
        avg.b = static_cast<unsigned char>(sumB / coveredCount);
        avg.a = 255;

        if (coveredCount == sampleCount) {
            
            result.color = avg;
        } else {
            float coverage = static_cast<float>(coveredCount) / sampleCount;
            result.color = mixColor(avg, bg[idx], coverage);
        }
        result.depth = minDepth;
    }

    out[idx] = result;
}

extern "C" void msaaResolveKernelLauncher(const void* sampleBuf, const void* bgColors,
                                           void* outPixels, int pixelCount, int sampleCount)
{
    int threads = 256;
    int blocks = (pixelCount + threads - 1) / threads;
    msaaResolveKernel<<<blocks, threads>>>((const SampleDevice*)sampleBuf,
                                           (const ColorDevice*)bgColors,
                                           (PixelDevice*)outPixels,
                                           pixelCount, sampleCount);
    cudaDeviceSynchronize();
} 