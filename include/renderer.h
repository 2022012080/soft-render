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

// 光源类型枚举
enum class LightType {
    POINT,      // 点光源
    DIRECTIONAL // 平面光源（方向光）
};

// 光源结构体
struct Light {
    LightType type;
    Vec3f position;     // 对于点光源是位置，对于平面光源是方向
    Vec3f color;
    float intensity;
    bool enabled;
    
    Light() : type(LightType::POINT), position(0, 0, 0), color(1, 1, 1), intensity(1.0f), enabled(true) {}
    Light(const Vec3f& pos, const Vec3f& col, float intens, LightType lightType = LightType::POINT) 
        : type(lightType), position(pos), color(col), intensity(intens), enabled(true) {}
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
    
    // 法线贴图
    std::shared_ptr<Texture> currentNormalMap;
    
    // 多光源系统
    std::vector<Light> lights;
    float ambientIntensity;
    
    // 新增：光照系数控制
    float diffuseStrength;    // 漫反射强度系数
    float specularStrength;   // 高光强度系数
    float ambientStrength;    // 环境光强度系数
    float shininess;          // 新增：高光指数（控制高光集中程度）
    
    // 新增：绘制控制开关
    bool m_drawTriangleEdges;
    bool m_drawLightRays;
    bool m_drawAxesAndGrid;  // 新增：坐标轴和网格线控制
    
    // 新增：纹理启用控制
    bool m_enableTexture;
    
    // 新增：法线贴图启用控制
    bool m_enableNormalMap;
    
    // 新增：BRDF 模型控制
    bool m_enableBRDF;            // BRDF 模型启用开关
    float m_roughness;            // 表面粗糙度 (0-1)
    float m_metallic;             // 金属度 (0-1)
    Vec3f m_F0;                   // 基础反射率（菲涅尔F0）
    
    // 新增：能量补偿相关参数
    bool m_enableEnergyCompensation;  // 能量补偿启用开关
    float m_energyCompensationScale;   // 能量补偿强度缩放因子
    
    // 新增：预计算的能量补偿查找表
    mutable std::vector<std::vector<float>> m_energyCompensationLUT;
    mutable bool m_lutInitialized;
    static constexpr int LUT_SIZE = 64;  // 查找表分辨率
    
    // 新增：位移着色器参数
    bool m_enableDisplacement;     // 位移着色器开关
    float m_displacementScale;     // 位移强度
    float m_displacementFrequency; // 位移频率
    float m_spineLength;          // 刺的长度
    float m_spineSharpness;       // 刺的锐利度
    
    // 新增：计算海胆刺位移
    Vec3f calculateSeaUrchinDisplacement(const Vec3f& position, const Vec3f& normal) const;
    
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
    
    // 设置法线贴图
    void setNormalMap(std::shared_ptr<Texture> normalMap) { currentNormalMap = normalMap; }
    
    // 设置多光源
    void addLight(const Light& light) { lights.push_back(light); }
    void setLight(int index, const Light& light);
    void removeLight(int index);
    void clearLights() { lights.clear(); }
    int getLightCount() const { return static_cast<int>(lights.size()); }
    const Light& getLight(int index) const;
    Light& getLight(int index);
    void setAmbientIntensity(float intensity) { ambientIntensity = intensity; }
    
    // 新增：光照系数控制方法
    void setDiffuseStrength(float strength) { diffuseStrength = strength; }
    void setSpecularStrength(float strength) { specularStrength = strength; }
    void setAmbientStrength(float strength) { ambientStrength = strength; }
    void setShininess(float value) { shininess = value; }  // 新增：设置高光指数
    float getDiffuseStrength() const { return diffuseStrength; }
    float getSpecularStrength() const { return specularStrength; }
    float getAmbientStrength() const { return ambientStrength; }
    float getShininess() const { return shininess; }  // 新增：获取高光指数
    
    // 新增：设置绘制控制开关
    void setDrawTriangleEdges(bool draw) { m_drawTriangleEdges = draw; }
    void setDrawLightRays(bool draw) { m_drawLightRays = draw; }
    void setDrawAxesAndGrid(bool draw) { m_drawAxesAndGrid = draw; }
    bool getDrawTriangleEdges() const { return m_drawTriangleEdges; }
    bool getDrawLightRays() const { return m_drawLightRays; }
    bool getDrawAxesAndGrid() const { return m_drawAxesAndGrid; }
    
    // 新增：纹理启用控制方法
    void setTextureEnabled(bool enabled) { m_enableTexture = enabled; }
    bool isTextureEnabled() const { return m_enableTexture; }
    
    // 新增：法线贴图启用控制方法
    void setNormalMapEnabled(bool enabled) { m_enableNormalMap = enabled; }
    bool isNormalMapEnabled() const { return m_enableNormalMap; }
    
    // 新增：BRDF 模型控制方法
    void setBRDFEnabled(bool enabled) { m_enableBRDF = enabled; }
    bool isBRDFEnabled() const { return m_enableBRDF; }
    void setRoughness(float roughness) { m_roughness = std::max(0.01f, std::min(1.0f, roughness)); }
    void setMetallic(float metallic) { m_metallic = std::max(0.0f, std::min(1.0f, metallic)); }
    void setF0(const Vec3f& f0) { m_F0 = f0; }
    float getRoughness() const { return m_roughness; }
    float getMetallic() const { return m_metallic; }
    Vec3f getF0() const { return m_F0; }
    
    // 新增：能量补偿控制方法
    void setEnergyCompensationEnabled(bool enabled) { m_enableEnergyCompensation = enabled; }
    bool isEnergyCompensationEnabled() const { return m_enableEnergyCompensation; }
    void setEnergyCompensationScale(float scale) { m_energyCompensationScale = std::max(0.0f, std::min(2.0f, scale)); }
    float getEnergyCompensationScale() const { return m_energyCompensationScale; }
    
    // 新增：位移着色器控制方法
    void setDisplacementEnabled(bool enabled) { m_enableDisplacement = enabled; }
    bool isDisplacementEnabled() const { return m_enableDisplacement; }
    void setDisplacementScale(float scale) { m_displacementScale = scale; }
    void setDisplacementFrequency(float freq) { m_displacementFrequency = freq; }
    void setSpineLength(float length) { m_spineLength = length; }
    void setSpineSharpness(float sharpness) { m_spineSharpness = sharpness; }
    float getDisplacementScale() const { return m_displacementScale; }
    float getDisplacementFrequency() const { return m_displacementFrequency; }
    float getSpineLength() const { return m_spineLength; }
    float getSpineSharpness() const { return m_spineSharpness; }
    
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
    
    // BRDF 相关计算函数
    Vec3f calculateBRDFLighting(const Vec3f& localPos, const Vec3f& localNormal, const Vec3f& baseColor);
    Vec3f calculateSingleLightBRDF(const Light& light, const Vec3f& localPos, const Vec3f& localNormal, const Vec3f& baseColor);
    
    // GGX 法线分布函数 (Trowbridge-Reitz)
    float distributionGGX(const Vec3f& N, const Vec3f& H, float roughness) const;
    
    // Smith 几何遮蔽函数的子函数
    float geometrySchlickGGX(float NdotV, float roughness) const;
    float geometrySmith(const Vec3f& N, const Vec3f& V, const Vec3f& L, float roughness) const;
    
    // Fresnel 方程 (Schlick 近似)
    Vec3f fresnelSchlick(float cosTheta, const Vec3f& F0) const;
    
    // 正面剔除
    bool isFrontFace(const Vec3f& v0, const Vec3f& v1, const Vec3f& v2);
    
    // 重心坐标计算
    Vec3f barycentric(const Vec2f& a, const Vec2f& b, const Vec2f& c, const Vec2f& p);
    
    // 更新法向量变换矩阵
    void updateNormalMatrix();
    
    // 绘制单个光源位置
    void drawSingleLightPosition(const Light& light, int lightIndex = 0);
    
    // 新增：能量补偿相关函数
    void initializeEnergyCompensationLUT() const;
    float lookupEnergyCompensation(float roughness, float cosTheta) const;
    float computeEnergyIntegral(float roughness, float cosTheta) const;
    Vec3f calculateEnergyCompensationTerm(const Vec3f& N, const Vec3f& V, const Vec3f& L, 
                                          float roughness, const Vec3f& F0) const;
}; 