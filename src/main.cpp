#include "renderer.h"
#include "model.h"
#include "texture.h"
#include "math.h"
#include <iostream>
#include <memory>

int main() {
    std::cout << "软光栅渲染器启动..." << std::endl;
    
    // 创建渲染器
    const int width = 800;
    const int height = 600;
    Renderer renderer(width, height);
    
    // 设置变换矩阵
    Matrix4x4 modelMatrix = Math::translate(Vec3f(0, 0, -5)) * 
                           Math::rotate(45, Vec3f(0, 1, 0)) * 
                           Math::scale(Vec3f(1, 1, 1));
    
    Matrix4x4 viewMatrix = Math::lookAt(
        Vec3f(0, 0, 5),    // 摄像机位置
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
    
    renderer.setModelMatrix(modelMatrix);
    renderer.setViewMatrix(viewMatrix);
    renderer.setProjectionMatrix(projectionMatrix);
    renderer.setViewportMatrix(viewportMatrix);
    
    // 设置光照
    renderer.setLightDirection(Vec3f(1, 1, 1));
    renderer.setLightColor(Vec3f(1, 1, 1));
    renderer.setAmbientIntensity(0.3f);
    
    // 创建默认纹理
    auto texture = std::make_shared<Texture>();
    texture->createDefault(256, 256);
    renderer.setTexture(texture);
    
    // 清空缓冲
    renderer.clear(Color(50, 50, 100));
    renderer.clearDepth();
    
    // 尝试加载OBJ文件
    Model model;
    if (model.loadFromFile("assets/cube.obj")) {
        std::cout << "成功加载OBJ模型，包含 " << model.getFaceCount() << " 个面" << std::endl;
        
        // 居中并缩放模型
        model.centerModel();
        model.scaleModel(2.0f);
        
        // 渲染模型
        std::cout << "渲染OBJ模型..." << std::endl;
        renderer.renderModel(model);
    } else {
        std::cout << "无法加载OBJ文件，使用内置立方体..." << std::endl;
        
        // 创建一个简单的立方体模型
        // 立方体的8个顶点
        std::vector<Vec3f> vertices = {
            Vec3f(-1, -1, -1), Vec3f(1, -1, -1), Vec3f(1, 1, -1), Vec3f(-1, 1, -1),
            Vec3f(-1, -1, 1), Vec3f(1, -1, 1), Vec3f(1, 1, 1), Vec3f(-1, 1, 1)
        };
        
        // 立方体的6个面（每个面2个三角形）
        std::vector<std::vector<int>> faces = {
            {0, 1, 2, 0, 2, 3},  // 前面
            {5, 4, 7, 5, 7, 6},  // 后面
            {4, 0, 3, 4, 3, 7},  // 左面
            {1, 5, 6, 1, 6, 2},  // 右面
            {3, 2, 6, 3, 6, 7},  // 上面
            {4, 5, 1, 4, 1, 0}   // 下面
        };
        
        // 手动创建立方体数据
        std::vector<Vertex> cubeVertices;
        
        for (const auto& face : faces) {
            for (int i = 0; i < 6; i += 2) {
                Vec3f v0 = vertices[face[i]];
                Vec3f v1 = vertices[face[i + 1]];
                Vec3f v2 = vertices[face[i + 2]];
                
                // 计算面法向量
                Vec3f edge1 = v1 - v0;
                Vec3f edge2 = v2 - v0;
                Vec3f normal = edge1.cross(edge2).normalize();
                
                // 添加三个顶点
                cubeVertices.push_back(Vertex(v0, normal, Vec2f(0, 0)));
                cubeVertices.push_back(Vertex(v1, normal, Vec2f(1, 0)));
                cubeVertices.push_back(Vertex(v2, normal, Vec2f(0, 1)));
            }
        }
        
        // 渲染立方体
        std::cout << "渲染内置立方体..." << std::endl;
        for (size_t i = 0; i < cubeVertices.size(); i += 3) {
            renderer.renderTriangle(cubeVertices[i], cubeVertices[i + 1], cubeVertices[i + 2]);
        }
    }
    
    // 保存结果
    std::cout << "保存渲染结果..." << std::endl;
    if (renderer.saveImage("output.bmp")) {
        std::cout << "渲染完成！结果已保存为 output.bmp" << std::endl;
        std::cout << "图像尺寸: " << width << "x" << height << std::endl;
    } else {
        std::cerr << "保存失败！" << std::endl;
        return -1;
    }
    
    return 0;
} 