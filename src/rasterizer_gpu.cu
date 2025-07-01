#include <cuda_runtime.h>
#include <math.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <stdio.h>
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

// Device-side float3 math utilities
__device__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ float3 operator*(const float3& a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}
__device__ float3 operator*(float b, const float3& a) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}
__device__ float3 operator/(const float3& a, float b) {
    return make_float3(a.x / b, a.y / b, a.z / b);
}
__device__ float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ float3 normalize(const float3& a) {
    float len = sqrtf(dot(a, a));
    if (len > 1e-6f) return a / len;
    return a;
}

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

__device__ bool atomicMinIfFloat(float* addr, float value) {
    int* addr_i = reinterpret_cast<int*>(addr);
    int old_i = *addr_i;
    while (value < __int_as_float(old_i)) {
        int assumed = old_i;
        int old_val = atomicCAS(addr_i, assumed, __float_as_int(value));
        if (old_val == assumed) {
            return true;
        }
        old_i = old_val;
    }
    return false;
}

__device__ float3 phongLighting(
    const float3& pos, const float3& normal, const float3& baseColor,
    const LightDevice* lights, int lightCount,
    const MaterialDevice& mat, const float3& viewPos, float ambientIntensity) {
    float3 ambient = make_float3(baseColor.x * mat.ambient[0], baseColor.y * mat.ambient[1], baseColor.z * mat.ambient[2]);
    ambient = ambient * ambientIntensity;
    float3 total = ambient;
    for (int i = 0; i < lightCount; ++i) {
        if (!lights[i].enabled) continue;
        float3 lightDir;
        float attenuation = 1.0f;
        if (lights[i].type == 1) {
            // Directional light
            lightDir = make_float3(-lights[i].pos[0], -lights[i].pos[1], -lights[i].pos[2]);
            lightDir = normalize(lightDir);
            attenuation = lights[i].intensity;
        } else {
            // Point light
            float3 lvec = make_float3(lights[i].pos[0], lights[i].pos[1], lights[i].pos[2]) - pos;
            float dist = sqrtf(dot(lvec, lvec));
            if (dist > 1e-6f) lightDir = lvec / dist; else lightDir = lvec;
            attenuation = lights[i].intensity / (dist * dist + 1e-4f);
        }
        float diff = fmaxf(dot(normal, lightDir), 0.0f);
        float3 diffuse = make_float3(baseColor.x * lights[i].color[0], baseColor.y * lights[i].color[1], baseColor.z * lights[i].color[2]);
        diffuse = diffuse * diff * attenuation;
        diffuse = make_float3(diffuse.x * mat.diffuse[0], diffuse.y * mat.diffuse[1], diffuse.z * mat.diffuse[2]);
        float3 viewDir = normalize(viewPos - pos);
        float3 reflectDir = normalize(2.0f * dot(normal, lightDir) * normal - lightDir);
        float spec = powf(fmaxf(dot(viewDir, reflectDir), 0.0f), mat.shininess);
        float3 specular = make_float3(lights[i].color[0], lights[i].color[1], lights[i].color[2]);
        specular = specular * spec * attenuation;
        specular = make_float3(specular.x * mat.specular[0], specular.y * mat.specular[1], specular.z * mat.specular[2]);
        total = total + diffuse + specular;
    }
    return total;
}

__device__ int g_pixelWriteCount = 0;

__device__ float3 mul(const float3& a, const float3& b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__global__ void rasterizeKernel(const VertexDevice* vertices, const int* indices,
                                int triCount, int width, int height,
                                PixelDevice* frameBuf,
                                const LightDevice* lights, int lightCount,
                                const MaterialDevice* mat, const float3 viewPos, float ambientIntensity) {
    int triId = blockIdx.x * blockDim.x + threadIdx.x;
    if (triId >= triCount) return;
    int i0 = indices[triId * 3 + 0];
    int i1 = indices[triId * 3 + 1];
    int i2 = indices[triId * 3 + 2];
    VertexDevice v0 = vertices[i0];
    VertexDevice v1 = vertices[i1];
    VertexDevice v2 = vertices[i2];
    int minX = max(0, (int)floorf(fminf(fminf(v0.x, v1.x), v2.x)));
    int maxX = min(width - 1, (int)ceilf(fmaxf(fmaxf(v0.x, v1.x), v2.x)));
    int minY = max(0, (int)floorf(fminf(fminf(v0.y, v1.y), v2.y)));
    int maxY = min(height - 1, (int)ceilf(fmaxf(fmaxf(v0.y, v1.y), v2.y)));
    float denom = ((v1.y - v2.y)*(v0.x - v2.x) + (v2.x - v1.x)*(v0.y - v2.y));
    if (fabsf(denom) < 1e-6f) return;
    for (int py = minY; py <= maxY; ++py) {
        for (int px = minX; px <= maxX; ++px) {
            float u = ((v1.y - v2.y)*((px+0.5f) - v2.x) + (v2.x - v1.x)*((py+0.5f) - v2.y)) / denom;
            float v = ((v2.y - v0.y)*((px+0.5f) - v2.x) + (v0.x - v2.x)*((py+0.5f) - v2.y)) / denom;
            float w = 1.0f - u - v;
            if (!(u >= 0 && v >= 0 && w >= 0)) continue;
            float depth = u * v0.z + v * v1.z + w * v2.z;
            int index = py * width + px;
            if (index < 0 || index >= width * height) continue;
            bool depthUpdated = atomicMinIfFloat(&frameBuf[index].depth, depth);
            if (depthUpdated) {
                float3 n0 = make_float3(v0.nx, v0.ny, v0.nz);
                float3 n1 = make_float3(v1.nx, v1.ny, v1.nz);
                float3 n2 = make_float3(v2.nx, v2.ny, v2.nz);
                float3 interpNormal = normalize(n0 * u + n1 * v + n2 * w);
                float3 p0 = make_float3(v0.x, v0.y, v0.z);
                float3 p1 = make_float3(v1.x, v1.y, v1.z);
                float3 p2 = make_float3(v2.x, v2.y, v2.z);
                float3 interpPos = p0 * u + p1 * v + p2 * w;
                float3 viewDir = normalize(viewPos - interpPos);
                float3 color = make_float3(0.0f, 0.0f, 0.0f);
                float3 matDiffuse = make_float3(mat->diffuse[0], mat->diffuse[1], mat->diffuse[2]);
                float3 matSpecular = make_float3(mat->specular[0], mat->specular[1], mat->specular[2]);
                float shininess = mat->shininess;
                float3 ambient = mul(matDiffuse, make_float3(0.1f, 0.1f, 0.1f));
                color = color + ambient;
                for (int li = 0; li < lightCount; ++li) {
                    float3 lightPos = make_float3(lights[li].pos[0], lights[li].pos[1], lights[li].pos[2]);
                    float3 lightColor = make_float3(lights[li].color[0], lights[li].color[1], lights[li].color[2]);
                    float3 lightDir;
                    float attenuation = 1.0f;
                    if (lights[li].type == 1) {
                        lightDir = normalize(make_float3(-lightPos.x, -lightPos.y, -lightPos.z));
                        attenuation = lights[li].intensity;
                    } else {
                        lightDir = normalize(lightPos - interpPos);
                        float dist = sqrtf(dot(lightPos - interpPos, lightPos - interpPos));
                        attenuation = lights[li].intensity / (dist * dist + 1e-4f);
                    }
                    float diff = fmaxf(dot(interpNormal, lightDir), 0.0f);
                    float3 reflectDir = normalize(2.0f * dot(interpNormal, lightDir) * interpNormal - lightDir);
                    float spec = powf(fmaxf(dot(viewDir, reflectDir), 0.0f), shininess);
                    float3 diffuse = diff * mul(lightColor, matDiffuse) * attenuation;
                    float3 specular = spec * mul(lightColor, matSpecular) * attenuation;
                    color = color + diffuse + specular;
                }
                color.x = fminf(fmaxf(color.x, 0.0f), 1.0f);
                color.y = fminf(fmaxf(color.y, 0.0f), 1.0f);
                color.z = fminf(fmaxf(color.z, 0.0f), 1.0f);
                unsigned char r = (unsigned char)(color.x * 255.0f);
                unsigned char g = (unsigned char)(color.y * 255.0f);
                unsigned char b = (unsigned char)(color.z * 255.0f);
                frameBuf[index].color = { r, g, b, 255 };
                atomicAdd(&g_pixelWriteCount, 1);
            }
        }
    }
}

extern "C" void launchRasterizeKernel(const void* d_vertices, const void* d_indices,
                                       int triCount, int width, int height,
                                       void* d_frameBuffer,
                                       const void* d_lights, int lightCount,
                                       const void* d_mat, const float3 viewPos, float ambientIntensity) {
    printf("[launchRasterizeKernel] sizeof(PixelDevice) (device) = %zu\n", sizeof(PixelDevice));
    int zero = 0;
    cudaMemcpyToSymbol(g_pixelWriteCount, &zero, sizeof(int));
    dim3 block(128);
    dim3 grid((triCount + block.x - 1) / block.x);
    rasterizeKernel<<<grid, block>>>((const VertexDevice*)d_vertices, (const int*)d_indices,
                                     triCount, width, height,
                                     (PixelDevice*)d_frameBuffer,
                                     (const LightDevice*)d_lights, lightCount,
                                     (const MaterialDevice*)d_mat, viewPos, ambientIntensity);
    cudaDeviceSynchronize();
    int pixelWriteCount = 0;
    cudaMemcpyFromSymbol(&pixelWriteCount, g_pixelWriteCount, sizeof(int));
    printf("[launchRasterizeKernel] pixelWriteCount = %d\n", pixelWriteCount);
} 