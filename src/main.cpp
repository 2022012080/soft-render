#include "window.h"
#include "vector_math.h"
#include <iostream>

#ifdef _WIN32
    #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
    #endif
    #include <windows.h>
#endif

// 声明在前面
Matrix4x4 getRotationBetweenVectors(const Vec3f& from, const Vec3f& to);

class Camera {
private:
    Vec3f initial_forward;
    Vec3f initial_up;
    Vec3f initial_right;
    Vec3f position;
    Vec3f current_forward;
    Vec3f current_up;
    float speed;
    
public:
    Camera() : speed(0.1f) {
        // 保存初始方向
        initial_forward = Vec3f(0, 0, -1);
        initial_up = Vec3f(0, 1, 0);
        initial_right = initial_forward.cross(initial_up);
        position = Vec3f(0, 0, 0);
        current_forward = initial_forward;
        current_up = initial_up;
    }

    Vec3f getCameraForward() const { return current_forward; }
    Vec3f getCameraUp() const { return current_up; }
    Vec3f getPosition() const { return position; }
    
    bool isKeyPressed(char key) {
        return (GetAsyncKeyState(key) & 0x8000) != 0;
    }

    void handleMovement(float deltaTime) {
        // 获取当前朝向
        Vec3f current_forward = getCameraForward();
        Vec3f current_up = getCameraUp();
        Vec3f current_right = current_forward.cross(current_up);
        
        // 计算移动向量（在初始坐标系中）
        Vec3f movement(0, 0, 0);
        if (isKeyPressed('W')) movement = movement + initial_forward * speed;
        if (isKeyPressed('S')) movement = movement - initial_forward * speed;
        if (isKeyPressed('D')) movement = movement + initial_right * speed;
        if (isKeyPressed('A')) movement = movement - initial_right * speed;
        
        // 计算当前朝向与初始朝向的旋转矩阵
        Matrix4x4 rotation = getRotationBetweenVectors(initial_forward, current_forward);
        
        // 将移动向量根据相机旋转进行变换
        Vec3f rotated_movement = rotation * movement;
        
        // 应用移动
        position = position + rotated_movement * deltaTime;
    }
};

Matrix4x4 getRotationBetweenVectors(const Vec3f& from, const Vec3f& to) {
    // 计算旋转轴
    Vec3f rotationAxis = from.cross(to).normalize();
    
    // 计算旋转角度
    float cosAngle = from.dot(to);
    float angle = acos(cosAngle);
    
    // 创建旋转矩阵
    return VectorMath::rotate(angle, rotationAxis);
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    std::cout << "Soft Renderer starting..." << std::endl;
    
    RenderWindow window(1800, 1440);
    
    if (!window.Initialize()) {
        std::cerr << "Window initialization failed!" << std::endl;
        return -1;
    }
    
    std::cout << "Window created successfully, starting GUI mode..." << std::endl;
    window.Run();
    
    return 0;
}

// 为了兼容性，保留原来的main函数作为备用
int main() {
    return WinMain(GetModuleHandle(nullptr), nullptr, GetCommandLineA(), SW_SHOWNORMAL);
} 