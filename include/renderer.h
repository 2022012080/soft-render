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

// 光源结构体
struct Light {
    Vec3f position;
    Vec3f color;
    float intensity;
    bool enabled;
    
    Light() : position(0, 0, 0), color(1, 1, 1), intensity(1.0f), enabled(true) {}
    Light(const Vec3f& pos, const Vec3f& col, float intens) 
        : position(pos), color(col), intensity(intens), enabled(true) {}
};

// 渲染器类
class Renderer {
private:
    std::vector<Pixel> frameBuffer;
    std::vector<float> depthBuffer;
    int width, height;
    
    // 超采样抗锯齿相关
    bool m_enableSSAA;
    int m_ssaaScale;  // 超采样倍数，通常为2或4
    std::vector<Pixel> m_highResFrameBuffer;
    std::vector<float> m_highResDepthBuffer;
    int m_highResWidth, m_highResHeight;
    
    // 变换矩阵
    Matrix4x4 modelMatrix;
    Matrix4x4 viewMatrix;
    Matrix4x4 projectionMatrix;
    Matrix4x4 viewportMatrix;
    Matrix4x4 normalMatrix;
    
    // 纹理
    std::shared_ptr<Texture> currentTexture;
    
    // 多光源系统
    std::vector<Light> lights;
    float ambientIntensity;
    
    // 新增：绘制控制开关
    bool m_drawTriangleEdges;
    bool m_drawLightRays;
    
public:
    Renderer(int w, int h);
    
    // 清空缓冲
    void clear(const Color& color = Color(0, 0, 0));
    void clearDepth();
    
    // 超采样抗锯齿控制
    void enableSSAA(bool enable, int scale = 4);
    void disableSSAA();
    bool isSSAAEnabled() const { return m_enableSSAA; }
    int getSSAAScale() const { return m_ssaaScale; }
    
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
    
    // 设置多光源
    void addLight(const Light& light) { lights.push_back(light); }
    void setLight(int index, const Light& light);
    void removeLight(int index);
    void clearLights() { lights.clear(); }
    int getLightCount() const { return lights.size(); }
    const Light& getLight(int index) const;
    Light& getLight(int index);
    void setAmbientIntensity(float intensity) { ambientIntensity = intensity; }
    
    // 新增：设置绘制控制开关
    void setDrawTriangleEdges(bool draw) { m_drawTriangleEdges = draw; }
    void setDrawLightRays(bool draw) { m_drawLightRays = draw; }
    bool getDrawTriangleEdges() const { return m_drawTriangleEdges; }
    bool getDrawLightRays() const { return m_drawLightRays; }
    
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
    
    // 新增：绘制线段
    void drawLine(const Vec3f& start, const Vec3f& end, const Color& color, float width = 1.0f);
    void drawLineHighRes(const Vec3f& start, const Vec3f& end, const Color& color, float width = 1.0f);
    
    // 新增：绘制坐标轴
    void drawAxes(float length = 2.0f);
    
    // 新增：绘制网格
    void drawGrid(float size = 5.0f, int divisions = 10);
    
    // 新增：绘制光源位置
    void drawLightPosition();
    
    // 新增：绘制所有光源位置
    void drawAllLightPositions();
    
    // 新增：绘制光线到模型顶点
    void drawLightRays(const Model& model);
    
private:
    // 顶点着色器
    struct ShaderVertex {
        Vec3f position;
        Vec3f normal;
        Vec2f texCoord;
        Vec3f worldPos;
        Vec3f localPos;      // 新增：本地坐标位置
        Vec3f localNormal;   // 新增：本地坐标法向量
    };
    
    // 片段着色器
    struct ShaderFragment {
        Vec3f position;
        Vec3f normal;
        Vec2f texCoord;
        Vec3f worldPos;
        Vec3f localPos;      // 新增：本地坐标位置
        Vec3f localNormal;   // 新增：本地坐标法向量
    };
    
    // 超采样相关私有方法
    void initializeSSAABuffers();
    void downsampleFromHighRes();
    Matrix4x4 createHighResViewportMatrix() const;
    
    // 渲染到高分辨率缓冲区
    void renderToHighRes(const Model& model);
    void renderTriangleHighRes(const Vertex& v0, const Vertex& v1, const Vertex& v2);
    void rasterizeTriangleHighRes(const ShaderVertex& v0, const ShaderVertex& v1, const ShaderVertex& v2);
    bool depthTestHighRes(int x, int y, float depth);
    void setPixelHighRes(int x, int y, const Color& color, float depth);
    
    // 顶点着色器
    ShaderVertex vertexShader(const Vertex& vertex);
    
    // 片段着色器
    Color fragmentShader(const ShaderFragment& fragment);
    
    // 光栅化三角形
    void rasterizeTriangle(const ShaderVertex& v0, const ShaderVertex& v1, const ShaderVertex& v2);
    
    // 绘制三角形边线
    void drawTriangleEdges(const ShaderVertex& v0, const ShaderVertex& v1, const ShaderVertex& v2);
    void drawTriangleEdgesHighRes(const ShaderVertex& v0, const ShaderVertex& v1, const ShaderVertex& v2);
    
    // 绘制扫描线
    void drawScanline(int y, int x1, int x2, const ShaderFragment& f1, const ShaderFragment& f2);
    
    // 插值片段数据
    ShaderFragment interpolateFragment(const ShaderFragment& f1, const ShaderFragment& f2, float t);
    
    // 深度测试
    bool depthTest(int x, int y, float depth);
    
    // 设置像素
    void setPixel(int x, int y, const Color& color, float depth);
    
    // 计算光照
    Vec3f calculateLighting(const Vec3f& localPos, const Vec3f& localNormal, const Vec3f& baseColor);
    
    // 计算单个光源的光照贡献
    Vec3f calculateSingleLight(const Light& light, const Vec3f& localPos, const Vec3f& localNormal, const Vec3f& baseColor);
    
    // 正面剔除
    bool isFrontFace(const Vec3f& v0, const Vec3f& v1, const Vec3f& v2);
    
    // 重心坐标计算
    Vec3f barycentric(const Vec2f& a, const Vec2f& b, const Vec2f& c, const Vec2f& p);
    
    // 更新法向量变换矩阵
    void updateNormalMatrix();
    
    // 绘制单个光源位置
    void drawSingleLightPosition(const Light& light, int lightIndex = 0);
}; 