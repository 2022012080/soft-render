#include "texture.h"
#include <fstream>
#include <iostream>
#include <algorithm>

bool Texture::loadFromFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Cannot open texture file: " << filename << std::endl;
        return false;
    }
    
    if (!parseBMPHeader(file)) {
        std::cerr << "Invalid BMP header: " << filename << std::endl;
        return false;
    }
    
    if (!readBMPPixels(file)) {
        std::cerr << "Failed to read BMP pixels: " << filename << std::endl;
        return false;
    }
    
    file.close();
    return true;
}

void Texture::createDefault(int w, int h) {
    width = w;
    height = h;
    pixels.resize(width * height);
    
    // 创建棋盘格纹理
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            bool checker = ((x / 32) + (y / 32)) % 2 == 0;
            Color color = checker ? Color(255, 255, 255) : Color(128, 128, 128);
            setPixel(x, y, color);
        }
    }
}

bool Texture::parseBMPHeader(std::ifstream& file) {
    // 读取BMP文件头
    char signature[2];
    file.read(signature, 2);
    if (signature[0] != 'B' || signature[1] != 'M') {
        return false;
    }
    
    // 跳过文件大小等字段
    file.seekg(18, std::ios::cur);
    
    // 读取宽度和高度
    file.read(reinterpret_cast<char*>(&width), 4);
    file.read(reinterpret_cast<char*>(&height), 4);
    
    // 跳过其他字段到像素数据
    file.seekg(28, std::ios::cur);
    
    return true;
}

bool Texture::readBMPPixels(std::ifstream& file) {
    pixels.resize(width * height);
    
    // BMP是倒置存储的，所以从底部开始读取
    for (int y = height - 1; y >= 0; y--) {
        for (int x = 0; x < width; x++) {
            unsigned char b, g, r;
            file.read(reinterpret_cast<char*>(&b), 1);
            file.read(reinterpret_cast<char*>(&g), 1);
            file.read(reinterpret_cast<char*>(&r), 1);
            
            setPixel(x, y, Color(r, g, b));
        }
        
        // 处理行对齐
        int padding = (4 - (width * 3) % 4) % 4;
        file.seekg(padding, std::ios::cur);
    }
    
    return true;
}

Color Texture::sample(float u, float v) const {
    if (!isValid()) {
        return Color(255, 0, 255); // 紫色表示错误
    }
    
    // 重复纹理
    u = u - std::floor(u);
    v = v - std::floor(v);
    
    // 转换为像素坐标
    float x = u * (width - 1);
    float y = v * (height - 1);
    
    // 双线性插值
    int x0 = static_cast<int>(x);
    int y0 = static_cast<int>(y);
    int x1 = std::min(x0 + 1, width - 1);
    int y1 = std::min(y0 + 1, height - 1);
    
    float fx = x - x0;
    float fy = y - y0;
    
    Color c00 = getPixel(x0, y0);
    Color c10 = getPixel(x1, y0);
    Color c01 = getPixel(x0, y1);
    Color c11 = getPixel(x1, y1);
    
    // 插值
    Color c0 = Color(
        static_cast<unsigned char>(c00.r * (1 - fx) + c10.r * fx),
        static_cast<unsigned char>(c00.g * (1 - fx) + c10.g * fx),
        static_cast<unsigned char>(c00.b * (1 - fx) + c10.b * fx)
    );
    
    Color c1 = Color(
        static_cast<unsigned char>(c01.r * (1 - fx) + c11.r * fx),
        static_cast<unsigned char>(c01.g * (1 - fx) + c11.g * fx),
        static_cast<unsigned char>(c01.b * (1 - fx) + c11.b * fx)
    );
    
    return Color(
        static_cast<unsigned char>(c0.r * (1 - fy) + c1.r * fy),
        static_cast<unsigned char>(c0.g * (1 - fy) + c1.g * fy),
        static_cast<unsigned char>(c0.b * (1 - fy) + c1.b * fy)
    );
}

Vec3f Texture::sampleVec3f(float u, float v) const {
    return sample(u, v).toVec3f();
}

Color Texture::getPixel(int x, int y) const {
    if (x < 0 || x >= width || y < 0 || y >= height) {
        return Color(0, 0, 0);
    }
    return pixels[getPixelIndex(x, y)];
}

void Texture::setPixel(int x, int y, const Color& color) {
    if (x < 0 || x >= width || y < 0 || y >= height) {
        return;
    }
    pixels[getPixelIndex(x, y)] = color;
}

int Texture::getPixelIndex(int x, int y) const {
    return y * width + x;
} 