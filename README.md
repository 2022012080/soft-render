# 软光栅渲染器

这是一个使用C++实现的软光栅渲染器，支持导入3D模型和纹理，并实现正面静态渲染。

## 功能特性

- **3D模型加载**: 支持OBJ格式模型文件
- **纹理映射**: 支持BMP格式纹理文件
- **软光栅渲染**: 完全CPU实现的渲染管线
- **正面剔除**: 自动剔除背向摄像机的面
- **深度缓冲**: Z-buffer深度测试
- **光照计算**: 环境光和漫反射光照
- **透视投影**: 3D到2D的透视变换
- **图像输出**: 渲染结果保存为BMP格式

## 项目结构

```
soft render/
├── CMakeLists.txt          # CMake构建文件
├── include/                # 头文件目录
│   ├── math.h             # 数学库（向量、矩阵）
│   ├── model.h            # 3D模型类
│   ├── texture.h          # 纹理类
│   └── renderer.h         # 渲染器类
├── src/                   # 源文件目录
│   ├── main.cpp           # 主程序
│   ├── math.cpp           # 数学库实现
│   ├── model.cpp          # 模型加载实现
│   ├── texture.cpp        # 纹理加载实现
│   └── renderer.cpp       # 渲染器实现
└── assets/                # 资源文件目录
    └── cube.obj           # 示例立方体模型
```

## 编译和运行

### 环境要求

- C++17 编译器
- CMake 3.10+
- OpenGL库（用于窗口系统）

### 编译步骤

1. 创建构建目录：
```bash
mkdir build
cd build
```

2. 配置项目：
```bash
cmake ..
```

3. 编译项目：
```bash
cmake --build .
```

4. 运行程序：
```bash
./SoftRenderer
```

## 使用方法

### 基本渲染

程序会自动创建一个立方体模型并进行渲染，结果保存为 `output.bmp`。

### 自定义模型

要加载自定义的OBJ模型：

```cpp
Model model;
if (model.loadFromFile("path/to/model.obj")) {
    model.centerModel();  // 居中模型
    model.scaleModel(1.0f);  // 缩放模型
    renderer.renderModel(model);
}
```

### 自定义纹理

要加载自定义的BMP纹理：

```cpp
auto texture = std::make_shared<Texture>();
if (texture->loadFromFile("path/to/texture.bmp")) {
    renderer.setTexture(texture);
}
```

### 变换矩阵

设置模型变换：

```cpp
Matrix4x4 modelMatrix = Math::translate(Vec3f(0, 0, -5)) * 
                       Math::rotate(45, Vec3f(0, 1, 0)) * 
                       Math::scale(Vec3f(1, 1, 1));
renderer.setModelMatrix(modelMatrix);
```

设置摄像机：

```cpp
Matrix4x4 viewMatrix = Math::lookAt(
    Vec3f(0, 0, 5),    // 摄像机位置
    Vec3f(0, 0, 0),    // 目标点
    Vec3f(0, 1, 0)     // 上方向
);
renderer.setViewMatrix(viewMatrix);
```

### 光照设置

```cpp
renderer.setLightDirection(Vec3f(1, 1, 1));  // 光照方向
renderer.setLightColor(Vec3f(1, 1, 1));      // 光照颜色
renderer.setAmbientIntensity(0.3f);          // 环境光强度
```

## 渲染管线

1. **顶点着色器**: 应用变换矩阵，将3D顶点转换到屏幕空间
2. **图元组装**: 将顶点组装成三角形
3. **正面剔除**: 剔除背向摄像机的三角形
4. **光栅化**: 将三角形转换为像素
5. **片段着色器**: 计算每个像素的颜色（纹理采样、光照计算）
6. **深度测试**: Z-buffer深度测试
7. **帧缓冲**: 将结果写入帧缓冲

## 技术细节

### 数学库

- **Vec2f/Vec3f**: 2D/3D向量类
- **Matrix4x4**: 4x4变换矩阵
- **数学函数**: 透视投影、视图变换、旋转、缩放等

### 模型加载

- 解析OBJ文件格式
- 支持顶点、法向量、纹理坐标
- 自动计算缺失的法向量

### 纹理系统

- 支持BMP格式纹理
- 双线性插值采样
- 纹理重复

### 渲染算法

- 重心坐标插值
- 扫描线光栅化
- 深度缓冲
- 环境光 + 漫反射光照模型

## 示例输出

程序会渲染一个带纹理的立方体，应用光照效果，并保存为BMP图像文件。

## 扩展功能

可以进一步扩展的功能：

- 支持更多模型格式（FBX、3DS等）
- 支持更多纹理格式（PNG、JPG等）
- 添加更多光照模型（高光、阴影等）
- 实现实时渲染窗口
- 添加后处理效果
- 支持动画和骨骼系统

## 许可证

本项目仅供学习和研究使用。 