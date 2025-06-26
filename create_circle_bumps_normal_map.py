#!/usr/bin/env python3
"""
生成圆形凸起法线贴图的脚本
为立方体创建8×8规律排列的圆形凸起效果
输出文件: circle_bumps_normal.bmp
"""

import numpy as np
import struct
import math

def create_bmp_header(width, height):
    """创建24位BMP文件头"""
    # 计算行填充
    row_size = ((width * 3 + 3) // 4) * 4
    image_size = row_size * height
    file_size = 54 + image_size
    
    # BMP文件头 (14字节)
    header = bytearray()
    header.extend(b'BM')  # 签名
    header.extend(struct.pack('<I', file_size))  # 文件大小
    header.extend(struct.pack('<H', 0))  # 保留字段1
    header.extend(struct.pack('<H', 0))  # 保留字段2
    header.extend(struct.pack('<I', 54))  # 数据偏移
    
    # DIB头 (40字节)
    header.extend(struct.pack('<I', 40))  # DIB头大小
    header.extend(struct.pack('<I', width))  # 宽度
    header.extend(struct.pack('<I', height))  # 高度
    header.extend(struct.pack('<H', 1))  # 颜色平面数
    header.extend(struct.pack('<H', 24))  # 每像素位数
    header.extend(struct.pack('<I', 0))  # 压缩方式
    header.extend(struct.pack('<I', image_size))  # 图像大小
    header.extend(struct.pack('<I', 0))  # X像素每米
    header.extend(struct.pack('<I', 0))  # Y像素每米
    header.extend(struct.pack('<I', 0))  # 使用的颜色数
    header.extend(struct.pack('<I', 0))  # 重要颜色数
    
    return header

def smooth_step(edge0, edge1, x):
    """平滑阶跃函数，用于创建平滑过渡"""
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)

def generate_circle_bumps_normal_map(width, height):
    """
    生成8×8圆形凸起的法线贴图
    在每个网格单元中创建一个圆形凸起
    """
    # 创建法线数组 (范围 -1 到 1)
    normals = np.zeros((height, width, 3), dtype=np.float32)
    
    # 圆形凸起参数
    grid_size = 8           # 8×8网格
    bump_height = 0.8       # 凸起高度
    bump_radius = 0.35      # 凸起半径（相对于网格单元大小）
    base_height = 0.1       # 基础表面高度
    falloff_sharpness = 2.0 # 边缘衰减锐度
    
    # 网格单元大小
    cell_size = 1.0 / grid_size
    
    for y in range(height):
        for x in range(width):
            # 标准化坐标到 [0, 1]
            u = x / width
            v = y / height
            
            # 计算在哪个网格单元中
            grid_x = int(u * grid_size)
            grid_y = int(v * grid_size)
            
            # 确保在边界内
            grid_x = min(grid_x, grid_size - 1)
            grid_y = min(grid_y, grid_size - 1)
            
            # 计算在网格单元内的相对位置 [0, 1]
            local_u = (u * grid_size) - grid_x
            local_v = (v * grid_size) - grid_y
            
            # 计算到网格单元中心的距离
            center_u = 0.5
            center_v = 0.5
            
            # 距离中心的向量
            dx = local_u - center_u
            dy = local_v - center_v
            
            # 到中心的距离
            distance = math.sqrt(dx * dx + dy * dy)
            
            # 标准化距离（相对于最大可能距离）
            max_distance = math.sqrt(0.5 * 0.5 + 0.5 * 0.5)  # 从中心到角的距离
            normalized_distance = distance / max_distance
            
            # 计算凸起高度
            if distance <= bump_radius:
                # 在凸起内部 - 使用余弦函数创建平滑的圆形凸起
                height_factor = math.cos((distance / bump_radius) * math.pi * 0.5)
                height_factor = pow(height_factor, falloff_sharpness)
                current_height = base_height + bump_height * height_factor
            else:
                # 在凸起外部 - 基础表面
                current_height = base_height
            
            # 计算法线
            if distance <= bump_radius and distance > 0:
                # 在凸起区域内，计算梯度
                # 使用数值微分方法计算梯度
                
                # 计算高度函数在当前位置的梯度
                epsilon = 1.0 / width  # 微小偏移
                
                # X方向梯度
                dx_plus = (local_u + epsilon) - center_u
                dy_plus = local_v - center_v
                dist_x_plus = math.sqrt(dx_plus * dx_plus + dy_plus * dy_plus)
                
                if dist_x_plus <= bump_radius:
                    height_x_plus = base_height + bump_height * pow(math.cos((dist_x_plus / bump_radius) * math.pi * 0.5), falloff_sharpness)
                else:
                    height_x_plus = base_height
                
                gradient_x = (height_x_plus - current_height) / epsilon
                
                # Y方向梯度
                dx_plus_y = local_u - center_u
                dy_plus_y = (local_v + epsilon) - center_v
                dist_y_plus = math.sqrt(dx_plus_y * dx_plus_y + dy_plus_y * dy_plus_y)
                
                if dist_y_plus <= bump_radius:
                    height_y_plus = base_height + bump_height * pow(math.cos((dist_y_plus / bump_radius) * math.pi * 0.5), falloff_sharpness)
                else:
                    height_y_plus = base_height
                
                gradient_y = (height_y_plus - current_height) / epsilon
                
                # 设置法线分量
                normals[y, x, 0] = -gradient_x * 0.5  # X分量（切线方向）
                normals[y, x, 1] = -gradient_y * 0.5  # Y分量（切线方向）
                normals[y, x, 2] = current_height     # Z分量（向上）
                
            else:
                # 在平坦区域，法线主要指向上方
                normals[y, x, 0] = 0.0
                normals[y, x, 1] = 0.0
                normals[y, x, 2] = current_height
            
            # 添加微小的随机细节纹理
            detail_noise = math.sin(u * 200) * math.cos(v * 200) * 0.02
            normals[y, x, 0] += detail_noise * math.cos(u * 100 + v * 100)
            normals[y, x, 1] += detail_noise * math.sin(u * 100 + v * 100)
    
    # 归一化所有法线向量
    for y in range(height):
        for x in range(width):
            normal = normals[y, x]
            length = math.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)
            if length > 0:
                normals[y, x] = normal / length
    
    # 将法线从 [-1, 1] 转换到 [0, 255] RGB值
    rgb_data = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            # 法线分量转换为RGB
            # X -> R, Y -> G, Z -> B
            rgb_data[y, x, 0] = int((normals[y, x, 0] + 1.0) * 127.5)  # R
            rgb_data[y, x, 1] = int((normals[y, x, 1] + 1.0) * 127.5)  # G
            rgb_data[y, x, 2] = int((normals[y, x, 2] + 1.0) * 127.5)  # B
    
    return rgb_data

def save_bmp(filename, rgb_data):
    """保存RGB数据为BMP文件"""
    height, width, _ = rgb_data.shape
    
    # 创建BMP头
    header = create_bmp_header(width, height)
    
    # 计算行填充
    row_size = ((width * 3 + 3) // 4) * 4
    padding = row_size - width * 3
    
    with open(filename, 'wb') as f:
        # 写入头部
        f.write(header)
        
        # 写入像素数据 (BMP是从下到上存储)
        for y in range(height - 1, -1, -1):
            for x in range(width):
                # BMP格式是BGR顺序
                b = rgb_data[y, x, 2]  # B
                g = rgb_data[y, x, 1]  # G
                r = rgb_data[y, x, 0]  # R
                f.write(bytes([b, g, r]))
            
            # 写入行填充
            for _ in range(padding):
                f.write(b'\x00')

def main():
    """主函数"""
    # 设置法线贴图尺寸
    width = 256
    height = 256
    
    print(f"生成8×8圆形凸起法线贴图 {width}x{height}...")
    
    # 生成圆形凸起法线贴图数据
    rgb_data = generate_circle_bumps_normal_map(width, height)
    
    # 保存为BMP文件
    output_file = "assets/circle_bumps_normal.bmp"
    save_bmp(output_file, rgb_data)
    
    print(f"圆形凸起法线贴图已保存到: {output_file}")
    print("圆形凸起法线贴图特征:")
    print("- 8×8规律排列的圆形凸起")
    print("- 每个凸起都有平滑的边缘过渡")
    print("- 凸起高度和半径经过优化")
    print("- 基础表面有微细纹理细节")
    print("- 适用于立方体各个面")
    print("\n使用方法:")
    print("1. 在程序中加载 cube1.obj")
    print("2. 加载 circle_bumps_normal.bmp 作为法线贴图")
    print("3. 启用法线贴图功能")
    print("4. 观察立方体表面的圆形凸起效果")
    print("\n参数说明:")
    print("- 网格大小: 8×8")
    print("- 凸起半径: 35% 网格单元大小")
    print("- 凸起高度: 可调节的平滑过渡")
    print("- 边缘衰减: 余弦函数平滑过渡")

if __name__ == "__main__":
    main() 