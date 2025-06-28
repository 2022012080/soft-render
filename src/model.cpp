#include "model.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

bool Model::loadFromFile(const std::string& filename) {
    if (!parseOBJ(filename)) {
        std::cerr << "Failed to load model: " << filename << std::endl;
        return false;
    }
    
    processVertices();
    return true;
}

bool Model::parseOBJ(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return false;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string token;
        iss >> token;
        
        if (token == "v") {
            // 顶点
            float x, y, z;
            iss >> x >> y >> z;
            vertices.push_back(Vec3f(x, y, z));
        }
        else if (token == "vn") {
            // 法向量
            float x, y, z;
            iss >> x >> y >> z;
            normals.push_back(Vec3f(x, y, z));
        }
        else if (token == "vt") {
            // 纹理坐标
            float u, v;
            iss >> u >> v;
            texCoords.push_back(Vec2f(u, v));
        }
        else if (token == "f") {
            // 面
            Face face;
            std::string vertex;
            
            for (int i = 0; i < 3; i++) {
                iss >> vertex;
                std::istringstream viss(vertex);
                std::string index;
                
                // 解析顶点索引
                std::getline(viss, index, '/');
                if (!index.empty()) {
                    face.vertices[i] = std::stoi(index) - 1;
                }
                
                // 解析纹理坐标索引
                if (std::getline(viss, index, '/') && !index.empty()) {
                    face.texCoords[i] = std::stoi(index) - 1;
                }
                
                // 解析法向量索引
                if (std::getline(viss, index, '/') && !index.empty()) {
                    face.normals[i] = std::stoi(index) - 1;
                }
            }
            
            faces.push_back(face);
        }
    }
    
    file.close();
    return true;
}

void Model::processVertices() {
    processedVertices.clear();
    
    for (const auto& face : faces) {
        for (int i = 0; i < 3; i++) {
            Vertex vertex;
            
            // 设置位置
            if (face.vertices[i] >= 0 && face.vertices[i] < vertices.size()) {
                Vec3f pos = vertices[face.vertices[i]];
                float tmp = pos.y; pos.y = pos.z; pos.z = tmp; // 交换y和z
                vertex.position = pos;
            }
            
            // 设置法向量
            if (face.normals[i] >= 0 && face.normals[i] < normals.size()) {
                Vec3f n = normals[face.normals[i]];
                float tmp = n.y; n.y = n.z; n.z = tmp; // 交换y和z
                vertex.normal = n;
            } else {
                // 如果没有法向量，计算面法向量
                Vec3f v0 = vertices[face.vertices[0]];
                Vec3f v1 = vertices[face.vertices[1]];
                Vec3f v2 = vertices[face.vertices[2]];
                Vec3f edge1 = v1 - v0;
                Vec3f edge2 = v2 - v0;
                Vec3f n = -(edge1.cross(edge2).normalize());
                float tmp = n.y; n.y = n.z; n.z = tmp; // 交换y和z
                vertex.normal = n;
            }
            
            // 设置纹理坐标
            if (face.texCoords[i] >= 0 && face.texCoords[i] < texCoords.size()) {
                vertex.texCoord = texCoords[face.texCoords[i]];
            } else {
                vertex.texCoord = Vec2f(0, 0);
            }
            
            processedVertices.push_back(vertex);
        }
    }
}

void Model::getFaceVertices(int faceIndex, Vertex& v0, Vertex& v1, Vertex& v2) const {
    if (faceIndex < 0 || faceIndex * 3 + 2 >= processedVertices.size()) {
        return;
    }
    
    v0 = processedVertices[faceIndex * 3];
    v1 = processedVertices[faceIndex * 3 + 1];
    v2 = processedVertices[faceIndex * 3 + 2];
}

void Model::getBoundingBox(Vec3f& min, Vec3f& max) const {
    if (vertices.empty()) {
        min = max = Vec3f(0, 0, 0);
        return;
    }
    
    min = max = vertices[0];
    for (const auto& vertex : vertices) {
        min.x = std::min(min.x, vertex.x);
        min.y = std::min(min.y, vertex.y);
        min.z = std::min(min.z, vertex.z);
        max.x = std::max(max.x, vertex.x);
        max.y = std::max(max.y, vertex.y);
        max.z = std::max(max.z, vertex.z);
    }
}

void Model::centerModel() {
    Vec3f min, max;
    getBoundingBox(min, max);
    Vec3f center = (min + max) * 0.5f;
    
    for (auto& vertex : vertices) {
        vertex = vertex - center;
    }
    
    // 重新处理顶点
    processVertices();
}

void Model::scaleModel(float scale) {
    for (auto& vertex : vertices) {
        vertex = vertex * scale;
    }
    
    // 重新处理顶点
    processVertices();
} 