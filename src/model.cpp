#include "model.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <unordered_map>
#include <array>

Model::Model() {}

bool Model::loadFromFile(const std::string& filename) {
    faceKa.clear(); faceKd.clear(); faceKs.clear(); faceKe.clear(); faceAlpha.clear();
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
    // 材质名到ka/kd/ks/ke/alpha
    struct MtlParams { Vec3f ka, kd, ks, ke; float alpha; MtlParams():ka(-1,-1,-1),kd(-1,-1,-1),ks(-1,-1,-1),ke(-1,-1,-1),alpha(1.0f){} };
    std::unordered_map<std::string, MtlParams> mtlParams;
    std::string currentMtl;
    std::string mtlFile;
    std::string objDir;
    size_t slash = filename.find_last_of("/\\");
    if (slash != std::string::npos) objDir = filename.substr(0, slash+1);
    // 先扫描OBJ
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string token;
        iss >> token;
        if (token == "mtllib") {
            iss >> mtlFile;
            if (!objDir.empty() && mtlFile.find(objDir) != 0) mtlFile = objDir + mtlFile;
            // 解析MTL
            std::ifstream mtl(mtlFile);
            if (mtl.is_open()) {
                std::string mline, mtlName;
                MtlParams params;
                while (std::getline(mtl, mline)) {
                    std::istringstream miss(mline);
                    std::string mtoken;
                    miss >> mtoken;
                    if (mtoken == "newmtl") {
                        if (!mtlName.empty()) mtlParams[mtlName] = params;
                        miss >> mtlName;
                        params = MtlParams();
                    } else if (mtoken == "Ka") {
                        float r=1,g=1,b=1; miss >> r >> g >> b; params.ka = Vec3f(r,g,b);
                    } else if (mtoken == "Kd") {
                        float r=1,g=1,b=1; miss >> r >> g >> b; params.kd = Vec3f(r,g,b);
                    } else if (mtoken == "Ks") {
                        float r=1,g=1,b=1; miss >> r >> g >> b; params.ks = Vec3f(r,g,b);
                    } else if (mtoken == "Ke") {
                        float r=0,g=0,b=0; miss >> r >> g >> b; params.ke = Vec3f(r,g,b);
                    } else if (mtoken == "d") {
                        float v; miss >> v; params.alpha = v;
                    }
                }
                if (!mtlName.empty()) mtlParams[mtlName] = params;
                mtl.close();
            }
        } else if (token == "usemtl") {
            iss >> currentMtl;
        } else if (token == "v") {
            float x, y, z; iss >> x >> y >> z;
            vertices.push_back(Vec3f(x, y, z));
        } else if (token == "vn") {
            float x, y, z; iss >> x >> y >> z;
            normals.push_back(Vec3f(x, y, z));
        } else if (token == "vt") {
            float u, v; iss >> u >> v;
            texCoords.push_back(Vec2f(u, v));
        } else if (token == "f") {
            Face face; std::string vertex;
            for (int i = 0; i < 3; i++) {
                iss >> vertex;
                std::istringstream viss(vertex);
                std::string index;
                std::getline(viss, index, '/');
                if (!index.empty()) face.vertices[i] = std::stoi(index) - 1;
                if (std::getline(viss, index, '/') && !index.empty()) face.texCoords[i] = std::stoi(index) - 1;
                if (std::getline(viss, index, '/') && !index.empty()) face.normals[i] = std::stoi(index) - 1;
            }
            faces.push_back(face);
            // 分配ka/kd/ks/ke/alpha
            Vec3f ka(-1,-1,-1),kd(-1,-1,-1),ks(-1,-1,-1),ke(-1,-1,-1); float alpha=1.0f;
            auto it = mtlParams.find(currentMtl);
            if (it != mtlParams.end()) {
                ka = it->second.ka; kd = it->second.kd; ks = it->second.ks; ke = it->second.ke; alpha = it->second.alpha;
            }
            faceKa.push_back(ka); faceKd.push_back(kd); faceKs.push_back(ks); faceKe.push_back(ke); faceAlpha.push_back(alpha);
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

Vec3f Model::getFaceKa(int faceIdx) const {
    if (faceIdx >= 0 && faceIdx < faceKa.size()) return faceKa[faceIdx];
    return Vec3f(-1,-1,-1);
}
Vec3f Model::getFaceKd(int faceIdx) const {
    if (faceIdx >= 0 && faceIdx < faceKd.size()) return faceKd[faceIdx];
    return Vec3f(-1,-1,-1);
}
Vec3f Model::getFaceKs(int faceIdx) const {
    if (faceIdx >= 0 && faceIdx < faceKs.size()) return faceKs[faceIdx];
    return Vec3f(-1,-1,-1);
}
Vec3f Model::getFaceKe(int faceIdx) const {
    if (faceIdx >= 0 && faceIdx < faceKe.size()) return faceKe[faceIdx];
    return Vec3f(-1,-1,-1);
}
float Model::getFaceAlpha(int faceIdx) const {
    if (faceIdx >= 0 && faceIdx < faceAlpha.size()) return faceAlpha[faceIdx];
    return 1.0f;
} 