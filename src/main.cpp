#include "renderer.h"
#include "model.h"
#include "texture.h"
#include "vector_math.h"
#include <iostream>
#include <memory>

int main() {
    std::cout << "Soft Raster Renderer Starting..." << std::endl;
    
    const int width = 800;
    const int height = 600;
    Renderer renderer(width, height);
    
    Matrix4x4 modelMatrix = VectorMath::translate(Vec3f(0, 0, -5)) * 
                           VectorMath::rotate(30, Vec3f(0, 1, 0)) * 
                           VectorMath::scale(Vec3f(0.5f, 0.5f, 0.5f));
    
    Matrix4x4 viewMatrix = VectorMath::lookAt(
        Vec3f(0, 0, 5),
        Vec3f(0, 0, 0),
        Vec3f(0, 1, 0)
    );
    
    Matrix4x4 projectionMatrix = VectorMath::perspective(20.0f, (float)width / height, 0.1f, 100.0f);
    
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
    
    renderer.setLightDirection(Vec3f(1, 1, 1));
    renderer.setLightColor(Vec3f(1, 1, 1));
    renderer.setAmbientIntensity(0.3f);
    
    auto texture = std::make_shared<Texture>();
    texture->createDefault(256, 256);
    renderer.setTexture(texture);
    
    renderer.clear(Color(50, 50, 100));
    renderer.clearDepth();
    
    Model model;
    if (model.loadFromFile("assets/cube.obj")) {
        std::cout << "Successfully loaded OBJ model with " << model.getFaceCount() << " faces" << std::endl;
        
        model.centerModel();
        model.scaleModel(0.5f);
        
        std::cout << "Rendering OBJ model..." << std::endl;
        renderer.renderModel(model);
    } else {
        std::cout << "Cannot load OBJ file, using built-in cube..." << std::endl;
        
        std::vector<Vec3f> vertices = {
            Vec3f(-1, -1, -1), Vec3f(1, -1, -1), Vec3f(1, 1, -1), Vec3f(-1, 1, -1),
            Vec3f(-1, -1, 1), Vec3f(1, -1, 1), Vec3f(1, 1, 1), Vec3f(-1, 1, 1)
        };
        
        std::vector<std::vector<int>> faces = {
            {0, 1, 2, 0, 2, 3},
            {5, 4, 7, 5, 7, 6},
            {4, 0, 3, 4, 3, 7},
            {1, 5, 6, 1, 6, 2},
            {3, 2, 6, 3, 6, 7},
            {4, 5, 1, 4, 1, 0}
        };
        
        std::vector<Vertex> cubeVertices;
        
        for (const auto& face : faces) {
            for (int i = 0; i < 6; i += 2) {
                Vec3f v0 = vertices[face[i]];
                Vec3f v1 = vertices[face[i + 1]];
                Vec3f v2 = vertices[face[i + 2]];
                
                Vec3f edge1 = v1 - v0;
                Vec3f edge2 = v2 - v0;
                Vec3f normal = edge1.cross(edge2).normalize();
                
                cubeVertices.push_back(Vertex(v0, normal, Vec2f(0, 0)));
                cubeVertices.push_back(Vertex(v1, normal, Vec2f(1, 0)));
                cubeVertices.push_back(Vertex(v2, normal, Vec2f(0, 1)));
            }
        }
        
        std::cout << "Rendering built-in cube..." << std::endl;
        for (size_t i = 0; i < cubeVertices.size(); i += 3) {
            renderer.renderTriangle(cubeVertices[i], cubeVertices[i + 1], cubeVertices[i + 2]);
        }
    }
    
    std::cout << "Saving render result..." << std::endl;
    if (renderer.saveImage("output.bmp")) {
        std::cout << "Rendering complete! Result saved as output.bmp" << std::endl;
        std::cout << "Image size: " << width << "x" << height << std::endl;
    } else {
        std::cerr << "Save failed!" << std::endl;
        return -1;
    }
    
    return 0;
} 