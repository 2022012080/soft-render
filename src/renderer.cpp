#include "renderer.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <cmath>

Renderer::Renderer(int w, int h) : width(w), height(h) {
    frameBuffer.resize(width * height);
    depthBuffer.resize(width * height);
    
    // 初始化点光源参数
    lightPosition = Vec3f(3, 3, 3);
    lightColor = Vec3f(1, 1, 1);
    lightIntensity = 10.0f;
    ambientIntensity = 0.2f;
    
    // 初始化法向量变换矩阵
    updateNormalMatrix();
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
    ShaderVertex sv0 = vertexShader(v0);
    ShaderVertex sv1 = vertexShader(v1);
    ShaderVertex sv2 = vertexShader(v2);
    
    // Convert to view space for proper face culling
    Vec3f worldPos0 = modelMatrix * v0.position;
    Vec3f worldPos1 = modelMatrix * v1.position;
    Vec3f worldPos2 = modelMatrix * v2.position;
    Vec3f viewPos0 = viewMatrix * worldPos0;
    Vec3f viewPos1 = viewMatrix * worldPos1;
    Vec3f viewPos2 = viewMatrix * worldPos2;
    
    if (!isFrontFace(viewPos0, viewPos1, viewPos2)) {
        return;
    }
    
    rasterizeTriangle(sv0, sv1, sv2);
}

Renderer::ShaderVertex Renderer::vertexShader(const Vertex& vertex) {
    ShaderVertex result;
    
    Vec3f worldPos = modelMatrix * vertex.position;
    Vec3f viewPos = viewMatrix * worldPos;
    Vec3f clipPos = projectionMatrix * viewPos;
    Vec3f screenPos = viewportMatrix * clipPos;
    
    // 变换法向量 - 使用法向量变换矩阵
    Vec3f transformedNormal = normalMatrix * vertex.normal;
    
    result.position = screenPos;
    result.normal = transformedNormal.normalize();
    result.texCoord = vertex.texCoord;
    result.worldPos = worldPos;
    result.localPos = vertex.position;        // 新增：保存本地坐标位置
    result.localNormal = vertex.normal;       // 新增：保存本地坐标法向量
    
    return result;
}

Color Renderer::fragmentShader(const ShaderFragment& fragment) {
    Vec3f baseColor(1, 1, 1);
    
    if (currentTexture && currentTexture->isValid()) {
        baseColor = currentTexture->sampleVec3f(fragment.texCoord.x, fragment.texCoord.y);
    }
    
    Vec3f finalColor = calculateLighting(fragment.localPos, fragment.localNormal, baseColor);
    
    return Color::fromVec3f(finalColor);
}

void Renderer::rasterizeTriangle(const ShaderVertex& v0, const ShaderVertex& v1, const ShaderVertex& v2) {
    int x0 = static_cast<int>(v0.position.x);
    int y0 = static_cast<int>(v0.position.y);
    int x1 = static_cast<int>(v1.position.x);
    int y1 = static_cast<int>(v1.position.y);
    int x2 = static_cast<int>(v2.position.x);
    int y2 = static_cast<int>(v2.position.y);
    
    int minX = std::max(0, std::min({x0, x1, x2}));
    int maxX = std::min(width - 1, std::max({x0, x1, x2}));
    int minY = std::max(0, std::min({y0, y1, y2}));
    int maxY = std::min(height - 1, std::max({y0, y1, y2}));
    
    for (int y = minY; y <= maxY; y++) {
        for (int x = minX; x <= maxX; x++) {
            Vec3f bary = barycentric(Vec2f(static_cast<float>(x0), static_cast<float>(y0)), 
                                   Vec2f(static_cast<float>(x1), static_cast<float>(y1)), 
                                   Vec2f(static_cast<float>(x2), static_cast<float>(y2)), 
                                   Vec2f(static_cast<float>(x), static_cast<float>(y)));
            
            if (bary.x >= 0 && bary.y >= 0 && bary.z >= 0) {
                float depth = bary.x * v0.position.z + bary.y * v1.position.z + bary.z * v2.position.z;
                
                if (depthTest(x, y, depth)) {
                    ShaderFragment fragment;
                    // 使用正确的重心坐标插值
                    fragment.position = v0.position * bary.x + v1.position * bary.y + v2.position * bary.z;
                    fragment.normal = v0.normal * bary.x + v1.normal * bary.y + v2.normal * bary.z;
                    fragment.texCoord = v0.texCoord * bary.x + v1.texCoord * bary.y + v2.texCoord * bary.z;
                    fragment.worldPos = v0.worldPos * bary.x + v1.worldPos * bary.y + v2.worldPos * bary.z;
                    fragment.localPos = v0.localPos * bary.x + v1.localPos * bary.y + v2.localPos * bary.z;
                    fragment.localNormal = v0.localNormal * bary.x + v1.localNormal * bary.y + v2.localNormal * bary.z;
                    
                    fragment.normal = fragment.normal.normalize();
                    fragment.localNormal = fragment.localNormal.normalize();
                    
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
    if (std::abs(denom) < 1e-6f) {
        return Vec3f(1, 0, 0);
    }
    
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

Vec3f Renderer::calculateLighting(const Vec3f& localPos, const Vec3f& localNormal, const Vec3f& baseColor) {
    // 环境光 - 增强强度，让场景更明亮
    Vec3f ambient = baseColor * (ambientIntensity * 0.6f); // 提高到60%
    
    // 在本地坐标系中计算光照（光源位置和表面位置都是本地坐标）
    Vec3f lightVector = lightPosition - localPos;
    float distance = lightVector.length();
    Vec3f lightDir = lightVector.normalize();
    
    // 计算衰减（距离的平方衰减）
    float attenuation = lightIntensity / (1.0f + 0.1f * distance + 0.01f * distance * distance);
    
    // 漫反射光照（Lambert模型） - 使用本地坐标系的法向量
    float diffuseStrength = std::max(0.0f, localNormal.dot(lightDir));
    Vec3f diffuse = baseColor * lightColor * diffuseStrength * attenuation;
    
    // 高光计算 - 只有当表面面向光源时才计算
    Vec3f specular(0, 0, 0);
    if (diffuseStrength > 0.0f) {  // 只有面向光源的表面才有高光
        // 计算视角方向 - 在本地坐标系中，摄像机位置需要变换
        Matrix4x4 invModelMatrix = VectorMath::inverse(modelMatrix);
        Vec3f localCameraPos = invModelMatrix * Vec3f(0, 0, 5);
        Vec3f viewDir = (localCameraPos - localPos).normalize();
        
        // 镜面反射光照（Phong模型）
        Vec3f reflectDir = localNormal * (2.0f * localNormal.dot(lightDir)) - lightDir;
        float specularStrength = std::pow(std::max(0.0f, viewDir.dot(reflectDir)), 32.0f);
        specular = lightColor * specularStrength * attenuation * 0.5f;
    }
    
    return ambient + diffuse + specular;
}

bool Renderer::isFrontFace(const Vec3f& v0, const Vec3f& v1, const Vec3f& v2) {
    Vec3f edge1 = v1 - v0;
    Vec3f edge2 = v2 - v0;
    Vec3f normal = edge1.cross(edge2);
    
    // In view space, camera looks down negative Z axis
    // Front faces have normals pointing towards camera (negative Z)
    return normal.z < 0;
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

void Renderer::updateNormalMatrix() {
    // 法向量变换矩阵是模型矩阵的逆转置矩阵
    // 对于仅包含旋转和均匀缩放的变换，可以直接使用模型矩阵的3x3部分
    normalMatrix = VectorMath::transpose(VectorMath::inverse(modelMatrix));
}

// 绘制线段 - 使用Bresenham算法
void Renderer::drawLine(const Vec3f& start, const Vec3f& end, const Color& color, float width) {
    // 将3D点转换到屏幕坐标
    Vec3f worldStart = modelMatrix * start;
    Vec3f worldEnd = modelMatrix * end;
    Vec3f viewStart = viewMatrix * worldStart;
    Vec3f viewEnd = viewMatrix * worldEnd;
    Vec3f clipStart = projectionMatrix * viewStart;
    Vec3f clipEnd = projectionMatrix * viewEnd;
    Vec3f screenStart = viewportMatrix * clipStart;
    Vec3f screenEnd = viewportMatrix * clipEnd;
    
    int x0 = static_cast<int>(screenStart.x);
    int y0 = static_cast<int>(screenStart.y);
    int x1 = static_cast<int>(screenEnd.x);
    int y1 = static_cast<int>(screenEnd.y);
    float z0 = screenStart.z;
    float z1 = screenEnd.z;
    
    // Bresenham直线算法
    int dx = abs(x1 - x0);
    int dy = abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;
    int err = dx - dy;
    
    int x = x0, y = y0;
    float totalDistance = std::sqrt(static_cast<float>(dx * dx + dy * dy));
    
    while (true) {
        // 计算当前点的深度值
        float currentDistance = std::sqrt(static_cast<float>((x - x0) * (x - x0) + (y - y0) * (y - y0)));
        float t = (totalDistance > 0) ? (currentDistance / totalDistance) : 0;
        float z = VectorMath::lerp(z0, z1, t);
        
        // 绘制像素（考虑线宽）
        int halfWidth = static_cast<int>(width / 2);
        for (int dx_offset = -halfWidth; dx_offset <= halfWidth; dx_offset++) {
            for (int dy_offset = -halfWidth; dy_offset <= halfWidth; dy_offset++) {
                int px = x + dx_offset;
                int py = y + dy_offset;
                if (px >= 0 && px < this->width && py >= 0 && py < this->height) {
                    if (depthTest(px, py, z)) {
                        // 如果颜色有alpha通道，进行alpha混合
                        if (color.a < 255) {
                            int index = py * this->width + px;
                            Color existingColor = frameBuffer[index].color;
                            
                            float alpha = color.a / 255.0f;
                            Color blendedColor(
                                static_cast<unsigned char>(color.r * alpha + existingColor.r * (1.0f - alpha)),
                                static_cast<unsigned char>(color.g * alpha + existingColor.g * (1.0f - alpha)),
                                static_cast<unsigned char>(color.b * alpha + existingColor.b * (1.0f - alpha)),
                                255
                            );
                            setPixel(px, py, blendedColor, z);
                        } else {
                            setPixel(px, py, color, z);
                        }
                    }
                }
            }
        }
        
        if (x == x1 && y == y1) break;
        
        int e2 = 2 * err;
        if (e2 > -dy) {
            err -= dy;
            x += sx;
        }
        if (e2 < dx) {
            err += dx;
            y += sy;
        }
    }
}

// 绘制坐标轴
void Renderer::drawAxes(float length) {
    Vec3f origin(0, 0, 0);
    
    // X轴 - 红色
    Vec3f xEnd(length, 0, 0);
    drawLine(origin, xEnd, Color(255, 0, 0), 2.0f);
    
    // Y轴 - 绿色
    Vec3f yEnd(0, length, 0);
    drawLine(origin, yEnd, Color(0, 255, 0), 2.0f);
    
    // Z轴 - 蓝色
    Vec3f zEnd(0, 0, length);
    drawLine(origin, zEnd, Color(0, 0, 255), 2.0f);
    
    // 在原点绘制一个小球表示原点
    Vec3f worldOrigin = modelMatrix * origin;
    Vec3f viewOrigin = viewMatrix * worldOrigin;
    Vec3f clipOrigin = projectionMatrix * viewOrigin;
    Vec3f screenOrigin = viewportMatrix * clipOrigin;
    
    int ox = static_cast<int>(screenOrigin.x);
    int oy = static_cast<int>(screenOrigin.y);
    float oz = screenOrigin.z;
    
    // 绘制原点标记（小圆圈）
    for (int dx = -3; dx <= 3; dx++) {
        for (int dy = -3; dy <= 3; dy++) {
            if (dx * dx + dy * dy <= 9) { // 半径为3的圆
                int px = ox + dx;
                int py = oy + dy;
                if (px >= 0 && px < this->width && py >= 0 && py < this->height) {
                    if (depthTest(px, py, oz)) {
                        setPixel(px, py, Color(255, 255, 255), oz);
                    }
                }
            }
        }
    }
}

// 绘制网格
void Renderer::drawGrid(float size, int divisions) {
    float step = size / divisions;
    Color gridColor(128, 128, 128, 128); // 半透明灰色
    
    // 绘制平行于X轴的线（在XY平面上）
    for (int i = -divisions; i <= divisions; i++) {
        float z = i * step;
        Vec3f start(-size, 0, z);
        Vec3f end(size, 0, z);
        drawLine(start, end, gridColor, 1.0f);
    }
    
    // 绘制平行于Z轴的线（在XY平面上）
    for (int i = -divisions; i <= divisions; i++) {
        float x = i * step;
        Vec3f start(x, 0, -size);
        Vec3f end(x, 0, size);
        drawLine(start, end, gridColor, 1.0f);
    }
}

// 绘制光源位置
void Renderer::drawLightPosition() {
    // 将光源位置从本地坐标系变换到世界坐标系
    Vec3f worldLightPos = modelMatrix * lightPosition;
    
    // 将世界坐标的光源位置转换到屏幕坐标
    Vec3f viewLightPos = viewMatrix * worldLightPos;
    Vec3f clipLightPos = projectionMatrix * viewLightPos;
    Vec3f screenLightPos = viewportMatrix * clipLightPos;
    
    int lx = static_cast<int>(screenLightPos.x);
    int ly = static_cast<int>(screenLightPos.y);
    float lz = screenLightPos.z;
    
    // 绘制光源标记（白色圆圈，比原点标记稍大）
    Color lightColor(255, 255, 255); // 白色
    int radius = 5; // 半径为5像素
    
    for (int dx = -radius; dx <= radius; dx++) {
        for (int dy = -radius; dy <= radius; dy++) {
            float distance = std::sqrt(static_cast<float>(dx * dx + dy * dy));
            if (distance <= radius) {
                int px = lx + dx;
                int py = ly + dy;
                if (px >= 0 && px < this->width && py >= 0 && py < this->height) {
                    if (depthTest(px, py, lz)) {
                        // 根据距离中心的远近调整亮度，创造渐变效果
                        float alpha = 1.0f - (distance / radius);
                        Color pixelColor(
                            static_cast<unsigned char>(255 * alpha),
                            static_cast<unsigned char>(255 * alpha),
                            static_cast<unsigned char>(255 * alpha)
                        );
                        setPixel(px, py, pixelColor, lz);
                    }
                }
            }
        }
    }
    
    // 在光源周围绘制一个小的十字标记，便于识别
    for (int i = -8; i <= 8; i++) {
        // 水平线
        int px = lx + i;
        int py = ly;
        if (px >= 0 && px < this->width && py >= 0 && py < this->height) {
            if (depthTest(px, py, lz)) {
                setPixel(px, py, lightColor, lz);
            }
        }
        
        // 垂直线
        px = lx;
        py = ly + i;
        if (px >= 0 && px < this->width && py >= 0 && py < this->height) {
            if (depthTest(px, py, lz)) {
                setPixel(px, py, lightColor, lz);
            }
        }
    }
}

// 绘制光线到模型顶点
void Renderer::drawLightRays(const Model& model) {
    const auto& vertices = model.getVertices();
    Color rayColor(255, 255, 150, 100); // 半透明浅黄色
    
    // 直接使用本地坐标系中的光源位置
    Vec3f localLightPos = lightPosition;
    
    // 遍历所有顶点，绘制从本地光源到每个顶点的连线
    std::vector<Vec3f> uniquePositions;
    
    // 收集唯一的顶点位置（避免重复绘制到同一个位置的光线）
    for (const auto& vertex : vertices) {
        // 使用模型本地坐标的顶点位置
        Vec3f localVertexPos = vertex.position;
        
        // 检查是否已经有相同位置的顶点
        bool isDuplicate = false;
        for (const auto& pos : uniquePositions) {
            Vec3f diff = localVertexPos - pos;
            if (diff.length() < 0.01f) { // 如果距离小于0.01，认为是重复顶点
                isDuplicate = true;
                break;
            }
        }
        
        if (!isDuplicate) {
            uniquePositions.push_back(localVertexPos);
        }
    }
    
    // 绘制从本地光源到每个唯一顶点位置的光线
    for (const auto& vertexPos : uniquePositions) {
        // 计算光线方向
        Vec3f lightToVertex = vertexPos - localLightPos;
        float distance = lightToVertex.length();
        
        // 只绘制一定距离内的光线，避免过长的线条
        if (distance < 10.0f) {
            // 在本地坐标系中绘制光线，drawLine会自动应用modelMatrix变换
            drawLine(localLightPos, vertexPos, rayColor, 1.0f);
        }
    }
} 