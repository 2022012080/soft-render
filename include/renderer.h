#pragma once
#include "vector_math.h"
#include "model.h"
#include "texture.h"
#include <vector>
#include <memory>

// 帧缓冲像素
struct Pixel {
    Color color;
    float depth;
    
    Pixel() : color(), depth(1.0f) {}
    Pixel(const Color& c, float d) : color(c), depth(d) {}
};

// 渲染器类
class Renderer {
private:
    std::vector<Pixel> frameBuffer;
    std::vector<float> depthBuffer;
    int width, height;
    
    // 变换矩阵
    Matrix4x4 modelMatrix;
    Matrix4x4 viewMatrix;
    Matrix4x4 projectionMatrix;
    Matrix4x4 viewportMatrix;
    Matrix4x4 normalMatrix;
    
    // 纹理
    std::shared_ptr<Texture> currentTexture;
    
    // 点光源参数
    Vec3f lightPosition;
    Vec3f lightColor;
    float lightIntensity;
    float ambientIntensity;
    
public:
    Renderer(int w, int h);
    
    // 清空缓冲
    void clear(const Color& color = Color(0, 0, 0));
    void clearDepth();
    
    // 设置变换矩阵
    void setModelMatrix(const Matrix4x4& matrix) { 
        modelMatrix = matrix; 
        updateNormalMatrix();
    }
    void setViewMatrix(const Matrix4x4& matrix) { viewMatrix = matrix; }
    void setProjectionMatrix(const Matrix4x4& matrix) { projectionMatrix = matrix; }
    void setViewportMatrix(const Matrix4x4& matrix) { viewportMatrix = matrix; }
    
    // 设置纹理
    void setTexture(std::shared_ptr<Texture> texture) { currentTexture = texture; }
    
    // 设置点光源
    void setLightPosition(const Vec3f& pos) { lightPosition = pos; }
    void setLightColor(const Vec3f& color) { lightColor = color; }
    void setLightIntensity(float intensity) { lightIntensity = intensity; }
    void setAmbientIntensity(float intensity) { ambientIntensity = intensity; }
    
    // 渲染模型
    void renderModel(const Model& model);
    
    // 渲染三角形
    void renderTriangle(const Vertex& v0, const Vertex& v1, const Vertex& v2);
    
    // 获取帧缓冲
    const std::vector<Pixel>& getFrameBuffer() const { return frameBuffer; }
    const std::vector<Color>& getColorBuffer() const;
    
    // 保存图像
    bool saveImage(const std::string& filename) const;
    
    // 获取尺寸
    int getWidth() const { return width; }
    int getHeight() const { return height; }
    
private:
    // 顶点着色器
    struct ShaderVertex {
        Vec3f position;
        Vec3f normal;
        Vec2f texCoord;
        Vec3f worldPos;
    };
    
    // 片段着色器
    struct ShaderFragment {
        Vec3f position;
        Vec3f normal;
        Vec2f texCoord;
        Vec3f worldPos;
    };
    
    // 顶点着色器
    ShaderVertex vertexShader(const Vertex& vertex);
    
    // 片段着色器
    Color fragmentShader(const ShaderFragment& fragment);
    
    // 光栅化三角形
    void rasterizeTriangle(const ShaderVertex& v0, const ShaderVertex& v1, const ShaderVertex& v2);
    
    // 绘制扫描线
    void drawScanline(int y, int x1, int x2, const ShaderFragment& f1, const ShaderFragment& f2);
    
    // 插值片段数据
    ShaderFragment interpolateFragment(const ShaderFragment& f1, const ShaderFragment& f2, float t);
    
    // 深度测试
    bool depthTest(int x, int y, float depth);
    
    // 设置像素
    void setPixel(int x, int y, const Color& color, float depth);
    
    // 计算光照
    Vec3f calculateLighting(const Vec3f& normal, const Vec3f& worldPos, const Vec3f& baseColor);
    
    // 正面剔除
    bool isFrontFace(const Vec3f& v0, const Vec3f& v1, const Vec3f& v2);
    
    // 重心坐标计算
    Vec3f barycentric(const Vec2f& a, const Vec2f& b, const Vec2f& c, const Vec2f& p);
    
    // 更新法向量变换矩阵
    void updateNormalMatrix();
}; 