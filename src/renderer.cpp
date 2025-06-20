#include "renderer.h"
#include <algorithm>
#include <fstream>
#include <iostream>

Renderer::Renderer(int w, int h) : width(w), height(h) {
    frameBuffer.resize(width * height);
    depthBuffer.resize(width * height);
    
    // 初始化光照
    lightDir = Vec3f(0, 1, 0);
    lightColor = Vec3f(1, 1, 1);
    ambientIntensity = 0.2f;
}

void Renderer::clear(const Color& color) {
    for (auto& pixel : frameBuffer) {
        pixel.color = color;
        pixel.depth = 1.0f;
    }
}

void Renderer::clearDepth() {
    for (auto& depth : depthBuffer) {
        depth = 1.0f;
    }
}

void Renderer::renderModel(const Model& model) {
    const auto& vertices = model.getVertices();
    
    for (size_t i = 0; i < vertices.size(); i += 3) {
        if (i + 2 < vertices.size()) {
            renderTriangle(vertices[i], vertices[i + 1], vertices[i + 2]);
        }
    }
}

void Renderer::renderTriangle(const Vertex& v0, const Vertex& v1, const Vertex& v2) {
    // 顶点着色器
    ShaderVertex sv0 = vertexShader(v0);
    ShaderVertex sv1 = vertexShader(v1);
    ShaderVertex sv2 = vertexShader(v2);
    
    // 正面剔除
    if (!isFrontFace(sv0.position, sv1.position, sv2.position)) {
        return;
    }
    
    // 光栅化
    rasterizeTriangle(sv0, sv1, sv2);
}

Renderer::ShaderVertex Renderer::vertexShader(const Vertex& vertex) {
    ShaderVertex result;
    
    // 应用变换矩阵
    Vec3f worldPos = modelMatrix * vertex.position;
    Vec3f viewPos = viewMatrix * worldPos;
    Vec3f clipPos = projectionMatrix * viewPos;
    Vec3f screenPos = viewportMatrix * clipPos;
    
    result.position = screenPos;
    result.normal = vertex.normal;
    result.texCoord = vertex.texCoord;
    result.worldPos = worldPos;
    
    return result;
}

Color Renderer::fragmentShader(const ShaderFragment& fragment) {
    Vec3f baseColor(1, 1, 1);
    
    // 纹理采样
    if (currentTexture && currentTexture->isValid()) {
        baseColor = currentTexture->sampleVec3f(fragment.texCoord.x, fragment.texCoord.y);
    }
    
    // 光照计算
    Vec3f finalColor = calculateLighting(fragment.normal, fragment.worldPos, baseColor);
    
    return Color::fromVec3f(finalColor);
}

void Renderer::rasterizeTriangle(const ShaderVertex& v0, const ShaderVertex& v1, const ShaderVertex& v2) {
    // 转换为屏幕坐标
    int x0 = static_cast<int>(v0.position.x);
    int y0 = static_cast<int>(v0.position.y);
    int x1 = static_cast<int>(v1.position.x);
    int y1 = static_cast<int>(v1.position.y);
    int x2 = static_cast<int>(v2.position.x);
    int y2 = static_cast<int>(v2.position.y);
    
    // 计算包围盒
    int minX = std::max(0, std::min({x0, x1, x2}));
    int maxX = std::min(width - 1, std::max({x0, x1, x2}));
    int minY = std::max(0, std::min({y0, y1, y2}));
    int maxY = std::min(height - 1, std::max({y0, y1, y2}));
    
    // 计算重心坐标
    for (int y = minY; y <= maxY; y++) {
        for (int x = minX; x <= maxX; x++) {
            Vec3f bary = barycentric(Vec2f(x0, y0), Vec2f(x1, y1), Vec2f(x2, y2), Vec2f(x, y));
            
            if (bary.x >= 0 && bary.y >= 0 && bary.z >= 0) {
                // 插值深度
                float depth = bary.x * v0.position.z + bary.y * v1.position.z + bary.z * v2.position.z;
                
                if (depthTest(x, y, depth)) {
                    // 插值其他属性
                    ShaderFragment fragment;
                    fragment.position = Math::lerp(v0.position, v1.position, bary.y) + 
                                      Math::lerp(v0.position, v2.position, bary.z);
                    fragment.normal = Math::lerp(v0.normal, v1.normal, bary.y) + 
                                     Math::lerp(v0.normal, v2.normal, bary.z);
                    fragment.texCoord = Math::lerp(v0.texCoord, v1.texCoord, bary.y) + 
                                       Math::lerp(v0.texCoord, v2.texCoord, bary.z);
                    fragment.worldPos = Math::lerp(v0.worldPos, v1.worldPos, bary.y) + 
                                       Math::lerp(v0.worldPos, v2.worldPos, bary.z);
                    
                    // 标准化法向量
                    fragment.normal = fragment.normal.normalize();
                    
                    // 片段着色器
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

Vec3f Renderer::calculateLighting(const Vec3f& normal, const Vec3f& worldPos, const Vec3f& baseColor) {
    // 环境光
    Vec3f ambient = baseColor * ambientIntensity;
    
    // 漫反射
    float diffuseIntensity = std::max(0.0f, normal.dot(lightDir));
    Vec3f diffuse = baseColor * lightColor * diffuseIntensity;
    
    return ambient + diffuse;
}

bool Renderer::isFrontFace(const Vec3f& v0, const Vec3f& v1, const Vec3f& v2) {
    Vec3f edge1 = v1 - v0;
    Vec3f edge2 = v2 - v0;
    Vec3f normal = edge1.cross(edge2);
    
    // 检查法向量是否朝向摄像机
    return normal.z > 0;
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