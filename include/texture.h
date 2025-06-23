#pragma once
#include "vector_math.h"
#include <vector>
#include <string>

// 颜色结构
struct Color {
    unsigned char r, g, b, a;
    
    Color() : r(0), g(0), b(0), a(255) {}
    Color(unsigned char r, unsigned char g, unsigned char b, unsigned char a = 255)
        : r(r), g(g), b(b), a(a) {}
    
    // 转换为浮点数
    Vec3f toVec3f() const {
        return Vec3f(r / 255.0f, g / 255.0f, b / 255.0f);
    }
    
    // 从浮点数创建
    static Color fromVec3f(const Vec3f& v) {
        return Color(
            static_cast<unsigned char>(VectorMath::clamp(v.x, 0.0f, 1.0f) * 255.0f),
            static_cast<unsigned char>(VectorMath::clamp(v.y, 0.0f, 1.0f) * 255.0f),
            static_cast<unsigned char>(VectorMath::clamp(v.z, 0.0f, 1.0f) * 255.0f)
        );
    }
};

// 纹理类
class Texture {
private:
    std::vector<Color> pixels;
    int width, height;
    
public:
    Texture() : width(0), height(0) {}
    
    // 加载BMP文件
    bool loadFromFile(const std::string& filename);
    
    // 创建默认纹理
    void createDefault(int w, int h);
    
    // 获取纹理尺寸
    int getWidth() const { return width; }
    int getHeight() const { return height; }
    
    // 采样纹理
    Color sample(float u, float v) const;
    Vec3f sampleVec3f(float u, float v) const;
    
    // 获取像素
    Color getPixel(int x, int y) const;
    
    // 设置像素
    void setPixel(int x, int y, const Color& color);
    
    // 检查纹理是否有效
    bool isValid() const { return width > 0 && height > 0 && !pixels.empty(); }
    
private:
    // 解析BMP文件头
    bool parseBMPHeader(std::ifstream& file);
    
    // 读取BMP像素数据
    bool readBMPPixels(std::ifstream& file);
    
    // 坐标转换
    int getPixelIndex(int x, int y) const;
}; 