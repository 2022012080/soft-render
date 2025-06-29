#pragma once
#include <cmath>
#include <iostream>

// 确保M_PI在MSVC中可用
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// 2D向量
struct Vec2f {
    float x, y;
    
    Vec2f() : x(0), y(0) {}
    Vec2f(float x, float y) : x(x), y(y) {}
    
    Vec2f operator+(const Vec2f& v) const { return Vec2f(x + v.x, y + v.y); }
    Vec2f operator-(const Vec2f& v) const { return Vec2f(x - v.x, y - v.y); }
    Vec2f operator*(float f) const { return Vec2f(x * f, y * f); }
    Vec2f operator*(const Vec2f& v) const { return Vec2f(x * v.x, y * v.y); }
    Vec2f operator/(float f) const { return Vec2f(x / f, y / f); }
    
    float norm() const { return std::sqrt(x * x + y * y); }
    Vec2f normalize() const { float n = norm(); return n > 0 ? *this / n : Vec2f(0, 0); }
    
    // 添加点积运算
    float dot(const Vec2f& v) const { return x * v.x + y * v.y; }
};

// 3D向量
struct Vec3f {
    float x, y, z;
    
    Vec3f() : x(0), y(0), z(0) {}
    Vec3f(float x, float y, float z) : x(x), y(y), z(z) {}
    
    Vec3f operator+(const Vec3f& v) const { return Vec3f(x + v.x, y + v.y, z + v.z); }
    Vec3f operator-(const Vec3f& v) const { return Vec3f(x - v.x, y - v.y, z - v.z); }
    Vec3f operator-() const { return Vec3f(-x, -y, -z); }  // 一元负号运算符
    Vec3f operator*(float f) const { return Vec3f(x * f, y * f, z * f); }
    Vec3f operator*(const Vec3f& v) const { return Vec3f(x * v.x, y * v.y, z * v.z); }
    Vec3f operator/(float f) const { return Vec3f(x / f, y / f, z / f); }
    
    // 添加复合赋值运算符
    Vec3f& operator+=(const Vec3f& v) { x += v.x; y += v.y; z += v.z; return *this; }
    Vec3f& operator-=(const Vec3f& v) { x -= v.x; y -= v.y; z -= v.z; return *this; }
    Vec3f& operator*=(float f) { x *= f; y *= f; z *= f; return *this; }
    Vec3f& operator/=(float f) { x /= f; y /= f; z /= f; return *this; }
    
    float norm() const { return std::sqrt(x * x + y * y + z * z); }
    float length() const { return norm(); }  // 别名方法，和norm()功能相同
    Vec3f normalize() const { float n = norm(); return n > 0 ? *this / n : Vec3f(0, 0, 0); }
    
    float dot(const Vec3f& v) const { return x * v.x + y * v.y + z * v.z; }
    Vec3f cross(const Vec3f& v) const {
        return Vec3f(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
    }
};

// 4D向量
struct Vec4f {
    float x, y, z, w;
    
    Vec4f() : x(0), y(0), z(0), w(1) {}
    Vec4f(float x, float y, float z, float w = 1.0f) : x(x), y(y), z(z), w(w) {}
    Vec4f(const Vec3f& v, float w = 1.0f) : x(v.x), y(v.y), z(v.z), w(w) {}
    
    Vec4f operator+(const Vec4f& v) const { return Vec4f(x + v.x, y + v.y, z + v.z, w + v.w); }
    Vec4f operator-(const Vec4f& v) const { return Vec4f(x - v.x, y - v.y, z - v.z, w - v.w); }
    Vec4f operator*(float f) const { return Vec4f(x * f, y * f, z * f, w * f); }
    Vec4f operator/(float f) const { return Vec4f(x / f, y / f, z / f, w / f); }
    
    float norm() const { return std::sqrt(x * x + y * y + z * z + w * w); }
    Vec4f normalize() const { float n = norm(); return n > 0 ? *this / n : Vec4f(0, 0, 0, 0); }
    
    float dot(const Vec4f& v) const { return x * v.x + y * v.y + z * v.z + w * v.w; }
    
    Vec3f xyz() const { return Vec3f(x, y, z); }
};

// 4x4矩阵
struct Matrix4x4 {
    float m[4][4];
    
    Matrix4x4() {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                m[i][j] = (i == j) ? 1.0f : 0.0f;
            }
        }
    }
    
    float& operator()(int i, int j) { return m[i][j]; }
    const float& operator()(int i, int j) const { return m[i][j]; }
    
    Matrix4x4 operator*(const Matrix4x4& mat) const {
        Matrix4x4 result;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                result.m[i][j] = 0;
                for (int k = 0; k < 4; k++) {
                    result.m[i][j] += m[i][k] * mat.m[k][j];
                }
            }
        }
        return result;
    }
    
    Vec3f operator*(const Vec3f& v) const {
        float w = m[3][0] * v.x + m[3][1] * v.y + m[3][2] * v.z + m[3][3];
        if (std::abs(w) < 1e-6f) w = 1.0f; // 避免除零
        return Vec3f(
            (m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z + m[0][3]) / w,
            (m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z + m[1][3]) / w,
            (m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z + m[2][3]) / w
        );
    }
    
    // 新增：矩阵与Vec4f相乘
    Vec4f operator*(const Vec4f& v) const {
        return Vec4f(
            m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z + m[0][3] * v.w,
            m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z + m[1][3] * v.w,
            m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z + m[2][3] * v.w,
            m[3][0] * v.x + m[3][1] * v.y + m[3][2] * v.z + m[3][3] * v.w
        );
    }
};

// 数学工具函数
namespace VectorMath {
    // 四元数类定义
    class Quaternion {
    public:
        float x, y, z, w;  // w是实部，(x,y,z)是虚部

        Quaternion() : x(0), y(0), z(0), w(1) {}
        Quaternion(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
        
        // 从欧拉角创建四元数（弧度）
        static Quaternion fromEulerAngles(float pitch, float yaw, float roll);
        // 从轴角创建四元数
        static Quaternion fromAxisAngle(const Vec3f& axis, float angle);
        
        Quaternion operator*(const Quaternion& q) const;
        Vec3f operator*(const Vec3f& v) const;
        
        Quaternion conjugate() const;
        Quaternion normalize() const;
        Matrix4x4 toMatrix() const;
        
        // 球面插值
        static Quaternion slerp(const Quaternion& q1, const Quaternion& q2, float t);
    };

    Matrix4x4 perspective(float fov, float aspect, float near, float far);
    Matrix4x4 lookAt(const Vec3f& eye, const Vec3f& center, const Vec3f& up);
    Matrix4x4 translate(const Vec3f& v);
    Matrix4x4 rotate(float angle, const Vec3f& axis);
    Matrix4x4 scale(const Vec3f& v);
    Matrix4x4 transpose(const Matrix4x4& m);
    Matrix4x4 inverse(const Matrix4x4& m);
    
    float clamp(float value, float min, float max);
    float lerp(float a, float b, float t);
    Vec3f lerp(const Vec3f& a, const Vec3f& b, float t);
    Vec2f lerp(const Vec2f& a, const Vec2f& b, float t);
}

inline Vec3f operator*(float f, const Vec3f& v) { return Vec3f(f * v.x, f * v.y, f * v.z); } 