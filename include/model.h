#pragma once
#include "math.h"
#include <vector>
#include <string>

// 顶点结构
struct Vertex {
    Vec3f position;    // 位置
    Vec3f normal;      // 法向量
    Vec2f texCoord;    // 纹理坐标
    
    Vertex() {}
    Vertex(const Vec3f& pos, const Vec3f& norm, const Vec2f& tex)
        : position(pos), normal(norm), texCoord(tex) {}
};

// 面结构
struct Face {
    int vertices[3];   // 三个顶点的索引
    int texCoords[3];  // 纹理坐标索引
    int normals[3];    // 法向量索引
    
    Face() {
        for (int i = 0; i < 3; i++) {
            vertices[i] = texCoords[i] = normals[i] = -1;
        }
    }
};

// 3D模型类
class Model {
private:
    std::vector<Vec3f> vertices;
    std::vector<Vec3f> normals;
    std::vector<Vec2f> texCoords;
    std::vector<Face> faces;
    
    // 处理后的顶点数据
    std::vector<Vertex> processedVertices;
    
public:
    Model() {}
    
    // 加载OBJ文件
    bool loadFromFile(const std::string& filename);
    
    // 获取处理后的顶点数据
    const std::vector<Vertex>& getVertices() const { return processedVertices; }
    
    // 获取面的数量
    size_t getFaceCount() const { return faces.size(); }
    
    // 获取指定面的顶点
    void getFaceVertices(int faceIndex, Vertex& v0, Vertex& v1, Vertex& v2) const;
    
    // 计算包围盒
    void getBoundingBox(Vec3f& min, Vec3f& max) const;
    
    // 居中模型
    void centerModel();
    
    // 缩放模型
    void scaleModel(float scale);
    
private:
    // 处理顶点数据
    void processVertices();
    
    // 解析OBJ文件
    bool parseOBJ(const std::string& filename);
}; 