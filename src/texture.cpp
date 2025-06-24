#include "texture.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>

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
    
    // 创建简单的渐变纹理，避免棋盘格插值问题
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // 简单的径向渐变
            float centerX = w / 2.0f;
            float centerY = h / 2.0f;
            float distance = std::sqrt((x - centerX) * (x - centerX) + (y - centerY) * (y - centerY));
            float maxDistance = std::sqrt(centerX * centerX + centerY * centerY);
            float intensity = 1.0f - (distance / maxDistance) * 0.5f;
            
            unsigned char value = static_cast<unsigned char>(intensity * 255);
            Color color(value, value, value);
            setPixel(x, y, color);
        }
    }
}

bool Texture::parseBMPHeader(std::ifstream& file) {
    // 重置文件指针到开始
    file.seekg(0, std::ios::beg);
    
    // 读取BMP文件头
    char signature[2];
    file.read(signature, 2);
    if (signature[0] != 'B' || signature[1] != 'M') {
        std::cerr << "不是有效的BMP文件" << std::endl;
        return false;
    }
    
    // 跳过文件大小(4字节)，保留字段(4字节)，读取数据偏移(4字节)
    file.seekg(4, std::ios::cur);  // 文件大小
    file.seekg(4, std::ios::cur);  // 保留字段
    
    uint32_t dataOffset;
    file.read(reinterpret_cast<char*>(&dataOffset), 4);
    
    // 跳过信息头大小(4字节)
    file.seekg(4, std::ios::cur);
    
    // 读取宽度和高度
    file.read(reinterpret_cast<char*>(&width), 4);
    file.read(reinterpret_cast<char*>(&height), 4);
    
    // 读取颜色平面数和每像素位数
    uint16_t planes, bitsPerPixel;
    file.read(reinterpret_cast<char*>(&planes), 2);
    file.read(reinterpret_cast<char*>(&bitsPerPixel), 2);
    
    if (bitsPerPixel != 24) {
        std::cerr << "只支持24位BMP文件，当前位深：" << bitsPerPixel << std::endl;
        return false;
    }
    
    // 跳转到像素数据位置
    file.seekg(dataOffset, std::ios::beg);
    
    std::cout << "BMP文件信息: " << width << "x" << height << ", " << bitsPerPixel << "位" << std::endl;
    
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
    
    // 转换为像素坐标 - 使用最近邻采样
    int x = static_cast<int>(u * (width - 1) + 0.5f);
    int y = static_cast<int>(v * (height - 1) + 0.5f);
    
    // 确保坐标在有效范围内
    x = std::max(0, std::min(x, width - 1));
    y = std::max(0, std::min(y, height - 1));
    
    // 直接返回最近的像素，不进行插值
    return getPixel(x, y);
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