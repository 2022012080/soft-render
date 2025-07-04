#pragma once
#include "vector_math.h"
#include "model.h"
#include "texture.h"
#include <vector>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>

// 帧缓冲像素
struct Pixel {
    Color color;
    float depth;
    
    Pixel() : color(), depth(1.0f) {}
    Pixel(const Color& c, float d) : color(c), depth(d) {}
};

// MSAA采样点结构
struct MSAASample {
    Color color;
    float depth;
    bool covered;  // 是否被三角形覆盖
    
    MSAASample() : color(), depth(1.0f), covered(false) {}
};

// MSAA像素，包含多个采样点
struct MSAAPixel {
    std::vector<MSAASample> samples;
    
    MSAAPixel(int sampleCount = 4) : samples(sampleCount) {}
    
    // 获取最终颜色（所有采样点的平均）
    Color getFinalColor(const Color& backgroundColor = Color(50, 50, 100)) const {
        if (samples.empty()) return backgroundColor;
        
        float r = 0, g = 0, b = 0;
        int coveredCount = 0;
        
        // 计算覆盖的采样点
        for (const auto& sample : samples) {
            if (sample.covered) {
                r += sample.color.r;
                g += sample.color.g;
                b += sample.color.b;
                coveredCount++;
            }
        }
        
        if (coveredCount == 0) return backgroundColor;
        
        // 如果所有采样点都被覆盖，返回采样点的平均值
        if (coveredCount == samples.size()) {
            return Color(
                static_cast<unsigned char>(r / coveredCount),
                static_cast<unsigned char>(g / coveredCount),
                static_cast<unsigned char>(b / coveredCount)
            );
        }
        
        // 部分覆盖：混合采样点颜色和背景色
        float coverage = static_cast<float>(coveredCount) / samples.size();
        float avgR = r / coveredCount;
        float avgG = g / coveredCount;
        float avgB = b / coveredCount;
        
        return Color(
            static_cast<unsigned char>(avgR * coverage + backgroundColor.r * (1.0f - coverage)),
            static_cast<unsigned char>(avgG * coverage + backgroundColor.g * (1.0f - coverage)),
            static_cast<unsigned char>(avgB * coverage + backgroundColor.b * (1.0f - coverage))
        );
    }
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

class ThreadPool {
public:
    ThreadPool(size_t numThreads);
    ~ThreadPool();
    void enqueue(std::function<void()> task);
    void waitAll();
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queueMutex;
    std::condition_variable condition;
    bool stop;
    size_t workingCount = 0;
    std::condition_variable doneCondition;
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
    
    // MSAA多重采样抗锯齿相关
    bool m_enableMSAA;
    int m_msaaSampleCount;  // 每像素采样点数，通常为4或8
    std::vector<MSAAPixel> m_msaaFrameBuffer;
    std::vector<Vec2f> m_samplePattern;  // 采样点位置模式
    
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
    Vec3f diffuseStrength;    // 漫反射强度系数
    Vec3f specularStrength;   // 高光强度系数
    Vec3f ambientStrength;    // 环境光强度系数
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
    
    // 新增：自发光参数
    bool m_enableEmission;       // 自发光启用开关
    float m_emissionStrength;    // 自发光强度
    Vec3f m_emissionColor;       // 自发光颜色
    
    // 新增：计算海胆刺位移
    Vec3f calculateSeaUrchinDisplacement(const Vec3f& position, const Vec3f& normal) const;
    
    std::unique_ptr<ThreadPool> m_threadPool;
    
    // 前向声明内部 ShaderVertex 以便后续成员函数参数可用
    struct ShaderVertex;
    
    // --- Shadow Mapping ---
    bool m_enableShadowMapping = false;           // 阴影映射开关
    bool m_shadowMapDirty = false;                // 阴影图是否需要重新生成
    int  m_shadowDebugCounter = 0;               // 调试帧计数器（切换后输出前三帧）
    int  m_shadowMapSize = 4096;                 // 阴影图分辨率
    std::vector<float> m_shadowDepthMap;         // 深度贴图
    Matrix4x4 m_lightViewMatrix;                 // 光源视图矩阵
    Matrix4x4 m_lightProjMatrix;                 // 光源投影矩阵
    Matrix4x4 m_lightViewProjMatrix;             // 组合矩阵
    
    // 调试统计
    mutable size_t m_debugShadowSampleCount = 0;
    mutable size_t m_debugShadowInShadowCount = 0;
    
public:
    virtual ~Renderer() = default;
    Renderer(int w, int h);
    
    // 新增：组合后的MVPV矩阵
    Matrix4x4 m_mvpvMatrix;

    // 清空缓冲
    void clear(const Color& color = Color(0, 0, 0));
    void clearDepth();
    
    // 超采样抗锯齿控制
    void enableSSAA(bool enable, int scale = 4);
    void disableSSAA();
    bool isSSAAEnabled() const { return m_enableSSAA; }
    int getSSAAScale() const { return m_ssaaScale; }
    
    // MSAA多重采样抗锯齿控制
    void enableMSAA(bool enable, int sampleCount = 4);
    void disableMSAA();
    bool isMSAAEnabled() const { return m_enableMSAA; }
    int getMSAASampleCount() const { return m_msaaSampleCount; }
    void resolveMSAAToFrameBuffer();
    
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
    void setDiffuseStrength(const Vec3f& strength) { diffuseStrength = strength; }
    void setSpecularStrength(const Vec3f& strength) { specularStrength = strength; }
    void setAmbientStrength(const Vec3f& strength) { ambientStrength = strength; }
    void setShininess(float value) { shininess = value; }  // 新增：设置高光指数
    Vec3f getDiffuseStrength() const { return diffuseStrength; }
    Vec3f getSpecularStrength() const { return specularStrength; }
    Vec3f getAmbientStrength() const { return ambientStrength; }
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
    
    // 新增：自发光控制方法
    void setEmissionEnabled(bool enabled) { m_enableEmission = enabled; }
    bool isEmissionEnabled() const { return m_enableEmission; }
    void setEmissionStrength(float strength) { m_emissionStrength = std::max(0.0f, strength); }
    float getEmissionStrength() const { return m_emissionStrength; }
    void setEmissionColor(const Vec3f& color) { m_emissionColor = color; }
    Vec3f getEmissionColor() const { return m_emissionColor; }
    
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
    virtual void renderModel(const Model& model);
    
    // 渲染三角形
    void renderTriangle(const Vertex& v0, const Vertex& v1, const Vertex& v2);
    
    // 获取帧缓冲
    const std::vector<Pixel>& getFrameBuffer() const { return frameBuffer; }
    
    // 获取颜色缓冲
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
    
    // SSAA 多重采样解析（将高分辨率缓冲区降采样到常规帧缓冲）
    void resolveSSAAToFrameBuffer();
    
    // 渲染到高分辨率缓冲区（带面索引/Alpha 支持）
    void renderTriangleHighResWithFaceIdx(const Vertex& v0, const Vertex& v1, const Vertex& v2, int faceIdx, const Model* pModel);
    void rasterizeTriangleHighResWithFaceIdx(const ShaderVertex& v0, const ShaderVertex& v1, const ShaderVertex& v2, int faceIdx, const Model* pModel);
    void setPixelHighResAlpha(int x, int y, const Color& src, float depth, float alpha);
    
    // 阴影控制接口
    void enableShadowMapping(bool enable);
    bool isShadowMappingEnabled() const { return m_enableShadowMapping; }
    void requestShadowMapRegeneration() { m_shadowMapDirty = true; } // 请求重新生成阴影图

    // --- 阴影实现函数 ---
    void generateShadowMap(const Model& model);
    bool isInShadow(const Vec3f& worldPos) const;

    // 工具：正交投影矩阵
    static Matrix4x4 orthographic(float left, float right, float bottom, float top, float near, float far);
    
    // 新增：保存深度图
    bool saveDepthMap(const std::string& filename) const;
    
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
    
    // MSAA相关私有方法
    void initializeMSAABuffers();
    void initializeSamplePattern();
    void renderTriangleMSAA(const Vertex& v0, const Vertex& v1, const Vertex& v2);
    void rasterizeTriangleMSAA(const ShaderVertex& v0, const ShaderVertex& v1, const ShaderVertex& v2, int faceIdx, const Model* pModel);
    bool isPointInTriangle(const Vec2f& p, const Vec2f& v0, const Vec2f& v1, const Vec2f& v2);
    
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
    float distributionGGX(const Vec3f& N, const Vec3f& H, float roughness, float NdotV) const;
    
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
    
    // 新增：面材质参数渲染支持
    void renderTriangleWithFaceIdx(const Vertex& v0, const Vertex& v1, const Vertex& v2, int faceIdx, const Model* pModel);
    void rasterizeTriangleWithFaceIdx(const ShaderVertex& v0, const ShaderVertex& v1, const ShaderVertex& v2, int faceIdx, const Model* pModel);
    Color fragmentShaderWithFaceIdx(const struct LocalShaderFragment& fragment, const Model* pModel);
    Vec3f calculateLightingWithParams(const Vec3f& localPos, const Vec3f& localNormal, const Vec3f& baseColor, const Vec3f& ka, const Vec3f& kd, const Vec3f& ks, const Vec3f& emissionColor);
    Vec3f calculateSingleLightWithParams(const Light& light, const Vec3f& localPos, const Vec3f& localNormal, const Vec3f& baseColor, const Vec3f& kd, const Vec3f& ks);
    void setPixelAlpha(int x, int y, const Color& src, float depth, float alpha);

#ifdef USE_CUDA
    void downsampleFromHighResCUDA();
#endif
}; 