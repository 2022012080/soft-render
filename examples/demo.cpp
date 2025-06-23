#include "../include/renderer.h"
#include "../include/model.h"
#include "../include/texture.h"
#include "../include/vector_math.h"
#include <iostream>
#include <memory>

void renderDemo() {
    std::cout << "软光栅渲染器演示程序" << std::endl;
    std::cout << "=====================" << std::endl;
    
    // 创建渲染器
    const int width = 1024;
    const int height = 768;
    Renderer renderer(width, height);
    
    // 设置摄像机
    Matrix4x4 viewMatrix = Math::lookAt(
        Vec3f(3, 2, 5),    // 摄像机位置
        Vec3f(0, 0, 0),    // 目标点
        Vec3f(0, 1, 0)     // 上方向
    );
    
    Matrix4x4 projectionMatrix = Math::perspective(45.0f, (float)width / height, 0.1f, 100.0f);
    
    // 视口变换矩阵
    Matrix4x4 viewportMatrix;
    viewportMatrix(0, 0) = width / 2.0f;
    viewportMatrix(1, 1) = height / 2.0f;
    viewportMatrix(2, 2) = 1.0f;
    viewportMatrix(0, 3) = width / 2.0f;
    viewportMatrix(1, 3) = height / 2.0f;
    
    renderer.setViewMatrix(viewMatrix);
    renderer.setProjectionMatrix(projectionMatrix);
    renderer.setViewportMatrix(viewportMatrix);
    
    // 设置光照
    renderer.setLightDirection(Vec3f(1, 1, 1).normalize());
    renderer.setLightColor(Vec3f(1, 1, 1));
    renderer.setAmbientIntensity(0.2f);
    
    // 注释掉纹理设置，让球体显示为纯白色
    // auto texture = std::make_shared<Texture>();
    // texture->createDefault(512, 512);
    // renderer.setTexture(texture);
    
    // 清空缓冲
    renderer.clear(Color(30, 30, 60));
    renderer.clearDepth();
    
    // 渲染多个立方体
    std::vector<Vec3f> positions = {
        Vec3f(-2, 0, 0),
        Vec3f(0, 0, 0),
        Vec3f(2, 0, 0),
        Vec3f(-1, 1.5, 0),
        Vec3f(1, 1.5, 0)
    };
    
    std::vector<float> rotations = {0, 30, 60, 45, 90};
    
    for (size_t i = 0; i < positions.size(); i++) {
        // 设置模型变换
        Matrix4x4 modelMatrix = Math::translate(positions[i]) * 
                               Math::rotate(rotations[i], Vec3f(0, 1, 0)) * 
                               Math::scale(Vec3f(0.8f, 0.8f, 0.8f));
        
        renderer.setModelMatrix(modelMatrix);
        
        // 创建立方体顶点
        std::vector<Vec3f> vertices = {
            Vec3f(-1, -1, -1), Vec3f(1, -1, -1), Vec3f(1, 1, -1), Vec3f(-1, 1, -1),
            Vec3f(-1, -1, 1), Vec3f(1, -1, 1), Vec3f(1, 1, 1), Vec3f(-1, 1, 1)
        };
        
        std::vector<std::vector<int>> faces = {
            {0, 1, 2, 0, 2, 3},  // 前面
            {5, 4, 7, 5, 7, 6},  // 后面
            {4, 0, 3, 4, 3, 7},  // 左面
            {1, 5, 6, 1, 6, 2},  // 右面
            {3, 2, 6, 3, 6, 7},  // 上面
            {4, 5, 1, 4, 1, 0}   // 下面
        };
        
        std::vector<Vertex> cubeVertices;
        
        for (const auto& face : faces) {
            for (int j = 0; j < 6; j += 2) {
                Vec3f v0 = vertices[face[j]];
                Vec3f v1 = vertices[face[j + 1]];
                Vec3f v2 = vertices[face[j + 2]];
                
                Vec3f edge1 = v1 - v0;
                Vec3f edge2 = v2 - v0;
                Vec3f normal = edge1.cross(edge2).normalize();
                
                cubeVertices.push_back(Vertex(v0, normal, Vec2f(0, 0)));
                cubeVertices.push_back(Vertex(v1, normal, Vec2f(1, 0)));
                cubeVertices.push_back(Vertex(v2, normal, Vec2f(0, 1)));
            }
        }
        
        // 渲染立方体
        for (size_t j = 0; j < cubeVertices.size(); j += 3) {
            renderer.renderTriangle(cubeVertices[j], cubeVertices[j + 1], cubeVertices[j + 2]);
        }
        
        std::cout << "渲染立方体 " << (i + 1) << "/" << positions.size() << std::endl;
    }
    
    // 保存结果
    std::cout << "保存渲染结果..." << std::endl;
    if (renderer.saveImage("demo_output.bmp")) {
        std::cout << "演示完成！结果已保存为 demo_output.bmp" << std::endl;
    } else {
        std::cerr << "保存失败！" << std::endl;
    }
}

int main() {
    renderDemo();
    return 0;
} 