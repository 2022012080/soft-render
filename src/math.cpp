#include "math.h"
#include <cmath>

namespace Math {
    
    Matrix4x4 perspective(float fov, float aspect, float near, float far) {
        Matrix4x4 result;
        float f = 1.0f / std::tan(fov * 0.5f * M_PI / 180.0f);
        
        result(0, 0) = f / aspect;
        result(1, 1) = f;
        result(2, 2) = (far + near) / (near - far);
        result(2, 3) = (2.0f * far * near) / (near - far);
        result(3, 2) = -1.0f;
        result(3, 3) = 0.0f;
        
        return result;
    }
    
    Matrix4x4 lookAt(const Vec3f& eye, const Vec3f& center, const Vec3f& up) {
        Matrix4x4 result;
        
        Vec3f f = (center - eye).normalize();
        Vec3f s = f.cross(up).normalize();
        Vec3f u = s.cross(f);
        
        result(0, 0) = s.x;
        result(0, 1) = s.y;
        result(0, 2) = s.z;
        result(1, 0) = u.x;
        result(1, 1) = u.y;
        result(1, 2) = u.z;
        result(2, 0) = -f.x;
        result(2, 1) = -f.y;
        result(2, 2) = -f.z;
        result(3, 0) = -s.dot(eye);
        result(3, 1) = -u.dot(eye);
        result(3, 2) = f.dot(eye);
        
        return result;
    }
    
    Matrix4x4 translate(const Vec3f& v) {
        Matrix4x4 result;
        result(0, 3) = v.x;
        result(1, 3) = v.y;
        result(2, 3) = v.z;
        return result;
    }
    
    Matrix4x4 rotate(float angle, const Vec3f& axis) {
        Matrix4x4 result;
        float c = std::cos(angle * M_PI / 180.0f);
        float s = std::sin(angle * M_PI / 180.0f);
        Vec3f a = axis.normalize();
        
        result(0, 0) = a.x * a.x * (1 - c) + c;
        result(0, 1) = a.x * a.y * (1 - c) - a.z * s;
        result(0, 2) = a.x * a.z * (1 - c) + a.y * s;
        result(1, 0) = a.y * a.x * (1 - c) + a.z * s;
        result(1, 1) = a.y * a.y * (1 - c) + c;
        result(1, 2) = a.y * a.z * (1 - c) - a.x * s;
        result(2, 0) = a.z * a.x * (1 - c) - a.y * s;
        result(2, 1) = a.z * a.y * (1 - c) + a.x * s;
        result(2, 2) = a.z * a.z * (1 - c) + c;
        
        return result;
    }
    
    Matrix4x4 scale(const Vec3f& v) {
        Matrix4x4 result;
        result(0, 0) = v.x;
        result(1, 1) = v.y;
        result(2, 2) = v.z;
        return result;
    }
    
    float clamp(float value, float min, float max) {
        if (value < min) return min;
        if (value > max) return max;
        return value;
    }
    
    float lerp(float a, float b, float t) {
        return a + (b - a) * t;
    }
    
    Vec3f lerp(const Vec3f& a, const Vec3f& b, float t) {
        return a + (b - a) * t;
    }
} 