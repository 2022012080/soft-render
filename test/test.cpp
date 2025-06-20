#include "../include/math.h"
#include "../include/model.h"
#include "../include/texture.h"
#include "../include/renderer.h"
#include <iostream>
#include <cassert>

void testMath() {
    std::cout << "测试数学库..." << std::endl;
    
    // 测试向量
    Vec3f v1(1, 2, 3);
    Vec3f v2(4, 5, 6);
    Vec3f sum = v1 + v2;
    assert(sum.x == 5 && sum.y == 7 && sum.z == 9);
    
    // 测试矩阵
    Matrix4x4 trans = Math::translate(Vec3f(1, 2, 3));
    assert(trans(0, 3) == 1 && trans(1, 3) == 2 && trans(2, 3) == 3);
    
    // 测试透视投影
    Matrix4x4 proj = Math::perspective(45.0f, 1.0f, 0.1f, 100.0f);
    assert(proj(3, 2) == -1.0f);
    
    std::cout << "数学库测试通过！" << std::endl;
}

void testTexture() {
    std::cout << "测试纹理系统..." << std::endl;
    
    Texture texture;
    texture.createDefault(64, 64);
    
    assert(texture.getWidth() == 64);
    assert(texture.getHeight() == 64);
    assert(texture.isValid());
    
    // 测试采样
    Color color = texture.sample(0.5f, 0.5f);
    assert(color.r > 0 || color.g > 0 || color.b > 0);
    
    std::cout << "纹理系统测试通过！" << std::endl;
}

void testRenderer() {
    std::cout << "测试渲染器..." << std::endl;
    
    Renderer renderer(100, 100);
    
    // 测试清空缓冲
    renderer.clear(Color(255, 0, 0));
    
    // 测试保存图像
    bool saved = renderer.saveImage("test_output.bmp");
    assert(saved);
    
    std::cout << "渲染器测试通过！" << std::endl;
}

void testModel() {
    std::cout << "测试模型加载..." << std::endl;
    
    Model model;
    bool loaded = model.loadFromFile("assets/cube.obj");
    
    if (loaded) {
        assert(model.getFaceCount() > 0);
        std::cout << "模型加载测试通过！" << std::endl;
    } else {
        std::cout << "模型文件不存在，跳过模型测试" << std::endl;
    }
}

int main() {
    std::cout << "软光栅渲染器测试程序" << std::endl;
    std::cout << "====================" << std::endl;
    
    try {
        testMath();
        testTexture();
        testRenderer();
        testModel();
        
        std::cout << std::endl;
        std::cout << "所有测试通过！" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "测试失败: " << e.what() << std::endl;
        return -1;
    }
} 