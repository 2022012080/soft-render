#include "vector_math.h"
#include <cmath>
#include <algorithm>

namespace VectorMath {
    
    Matrix4x4 perspective(float fov, float aspect, float near, float far) {
        Matrix4x4 result;
        float f = 1.0f / std::tan(fov * 0.5f * static_cast<float>(M_PI) / 180.0f);
        
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
        float c = std::cos(angle * static_cast<float>(M_PI) / 180.0f);
        float s = std::sin(angle * static_cast<float>(M_PI) / 180.0f);
        Vec3f a = axis.normalize();
        
        result(0, 0) = a.x * a.x * (1.0f - c) + c;
        result(0, 1) = a.x * a.y * (1.0f - c) - a.z * s;
        result(0, 2) = a.x * a.z * (1.0f - c) + a.y * s;
        result(1, 0) = a.y * a.x * (1.0f - c) + a.z * s;
        result(1, 1) = a.y * a.y * (1.0f - c) + c;
        result(1, 2) = a.y * a.z * (1.0f - c) - a.x * s;
        result(2, 0) = a.z * a.x * (1.0f - c) - a.y * s;
        result(2, 1) = a.z * a.y * (1.0f - c) + a.x * s;
        result(2, 2) = a.z * a.z * (1.0f - c) + c;
        
        return result;
    }
    
    Matrix4x4 scale(const Vec3f& v) {
        Matrix4x4 result;
        result(0, 0) = v.x;
        result(1, 1) = v.y;
        result(2, 2) = v.z;
        return result;
    }
    
    Matrix4x4 transpose(const Matrix4x4& m) {
        Matrix4x4 result;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                result(i, j) = m(j, i);
            }
        }
        return result;
    }
    
    Matrix4x4 inverse(const Matrix4x4& m) {
        // 简化版本：对于仅包含旋转和平移的矩阵
        // 对于更复杂的情况，需要完整的高斯消元法
        Matrix4x4 result;
        
        // 提取3x3旋转部分的转置（即逆）
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                result(i, j) = m(j, i);
            }
        }
        
        // 计算平移部分的逆
        Vec3f translation(-m(0, 3), -m(1, 3), -m(2, 3));
        Vec3f invTranslation(
            result(0, 0) * translation.x + result(0, 1) * translation.y + result(0, 2) * translation.z,
            result(1, 0) * translation.x + result(1, 1) * translation.y + result(1, 2) * translation.z,
            result(2, 0) * translation.x + result(2, 1) * translation.y + result(2, 2) * translation.z
        );
        
        result(0, 3) = invTranslation.x;
        result(1, 3) = invTranslation.y;
        result(2, 3) = invTranslation.z;
        result(3, 3) = 1.0f;
        
        return result;
    }
    
    float clamp(float value, float min, float max) {
        return std::min(std::max(value, min), max);
    }
    
    float lerp(float a, float b, float t) {
        return a + (b - a) * t;
    }
    
    Vec3f lerp(const Vec3f& a, const Vec3f& b, float t) {
        return a + (b - a) * t;
    }
    
    Vec2f lerp(const Vec2f& a, const Vec2f& b, float t) {
        return a + (b - a) * t;
    }

    // 四元数实现
    Quaternion Quaternion::fromEulerAngles(float pitch, float yaw, float roll) {
        // 将角度转换为弧度
        float cy = cos(yaw * 0.5f);
        float sy = sin(yaw * 0.5f);
        float cp = cos(pitch * 0.5f);
        float sp = sin(pitch * 0.5f);
        float cr = cos(roll * 0.5f);
        float sr = sin(roll * 0.5f);

        Quaternion q;
        q.w = cr * cp * cy + sr * sp * sy;
        q.x = sr * cp * cy - cr * sp * sy;
        q.y = cr * sp * cy + sr * cp * sy;
        q.z = cr * cp * sy - sr * sp * cy;
        
        return q;
    }

    Quaternion Quaternion::fromAxisAngle(const Vec3f& axis, float angle) {
        float halfAngle = angle * 0.5f;
        float s = sin(halfAngle);
        
        Quaternion q;
        q.w = cos(halfAngle);
        q.x = axis.x * s;
        q.y = axis.y * s;
        q.z = axis.z * s;
        
        return q;
    }

    Quaternion Quaternion::operator*(const Quaternion& q) const {
        return Quaternion(
            w * q.x + x * q.w + y * q.z - z * q.y,
            w * q.y - x * q.z + y * q.w + z * q.x,
            w * q.z + x * q.y - y * q.x + z * q.w,
            w * q.w - x * q.x - y * q.y - z * q.z
        );
    }

    Vec3f Quaternion::operator*(const Vec3f& v) const {
        // 将向量转换为纯四元数
        Quaternion p(v.x, v.y, v.z, 0);
        
        // q * p * q^-1
        Quaternion result = (*this) * p * this->conjugate();
        return Vec3f(result.x, result.y, result.z);
    }

    Quaternion Quaternion::conjugate() const {
        return Quaternion(-x, -y, -z, w);
    }

    Quaternion Quaternion::normalize() const {
        float len = sqrt(x * x + y * y + z * z + w * w);
        if (len > 0) {
            float invLen = 1.0f / len;
            return Quaternion(x * invLen, y * invLen, z * invLen, w * invLen);
        }
        return *this;
    }

    Matrix4x4 Quaternion::toMatrix() const {
        Matrix4x4 result;
        float xx = x * x;
        float xy = x * y;
        float xz = x * z;
        float xw = x * w;
        float yy = y * y;
        float yz = y * z;
        float yw = y * w;
        float zz = z * z;
        float zw = z * w;

        result(0, 0) = 1 - 2 * (yy + zz);
        result(0, 1) = 2 * (xy - zw);
        result(0, 2) = 2 * (xz + yw);
        
        result(1, 0) = 2 * (xy + zw);
        result(1, 1) = 1 - 2 * (xx + zz);
        result(1, 2) = 2 * (yz - xw);
        
        result(2, 0) = 2 * (xz - yw);
        result(2, 1) = 2 * (yz + xw);
        result(2, 2) = 1 - 2 * (xx + yy);

        return result;
    }

    Quaternion Quaternion::slerp(const Quaternion& q1, const Quaternion& q2, float t) {
        // 确保t在[0,1]范围内
        t = clamp(t, 0.0f, 1.0f);
        
        // 计算四元数点积
        float dot = q1.x * q2.x + q1.y * q2.y + q1.z * q2.z + q1.w * q2.w;
        
        // 如果点积为负，取q2的负值以获得较短的路径
        Quaternion target = q2;
        if (dot < 0) {
            dot = -dot;
            target = Quaternion(-q2.x, -q2.y, -q2.z, -q2.w);
        }
        
        // 如果四元数非常接近，使用线性插值
        if (dot > 0.9995f) {
            Quaternion result(
                q1.x + (target.x - q1.x) * t,
                q1.y + (target.y - q1.y) * t,
                q1.z + (target.z - q1.z) * t,
                q1.w + (target.w - q1.w) * t
            );
            return result.normalize();
        }
        
        // 执行球面插值
        float theta = acos(dot);
        float sinTheta = sin(theta);
        float s1 = sin((1 - t) * theta) / sinTheta;
        float s2 = sin(t * theta) / sinTheta;
        
        return Quaternion(
            q1.x * s1 + target.x * s2,
            q1.y * s1 + target.y * s2,
            q1.z * s1 + target.z * s2,
            q1.w * s1 + target.w * s2
        ).normalize();
    }
} 