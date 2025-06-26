#!/usr/bin/env python3
"""
生成砖墙样式法线贴图的脚本
为立方体创建逼真的砖墙表面效果
输出文件: brick_normal.bmp
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

def generate_brick_normal_map(width, height):
    """
    生成砖墙样式的法线贴图
    模拟真实砖墙的凹凸效果
    """
    # 创建法线数组 (范围 -1 到 1)
    normals = np.zeros((height, width, 3), dtype=np.float32)
    
    # 砖墙参数
    brick_width = 0.25    # 砖块宽度比例
    brick_height = 0.125  # 砖块高度比例
    mortar_width = 0.02   # 砂浆宽度
    brick_depth = 0.3     # 砖块深度（法线Z分量的影响）
    mortar_depth = -0.8   # 砂浆深度（凹陷）
    
    # 砖块表面纹理参数
    brick_roughness = 0.15  # 砖块表面粗糙度
    
    for y in range(height):
        for x in range(width):
            # 标准化坐标到 [0, 1]
            u = x / width
            v = y / height
            
            # 计算砖块网格位置
            # 偶数行和奇数行错位排列（典型砖墙模式）
            row = int(v / brick_height)
            col_offset = 0.5 * brick_width if row % 2 == 1 else 0.0
            
            # 调整u坐标以实现错位
            u_adjusted = (u + col_offset) % 1.0
            
            # 计算在砖块网格中的相对位置
            brick_u = (u_adjusted % brick_width) / brick_width
            brick_v = (v % brick_height) / brick_height
            
            # 判断是否在砂浆区域
            mortar_threshold = mortar_width / brick_width
            is_mortar_u = brick_u < mortar_threshold or brick_u > (1.0 - mortar_threshold)
            
            mortar_threshold_v = mortar_width / brick_height
            is_mortar_v = brick_v < mortar_threshold_v or brick_v > (1.0 - mortar_threshold_v)
            
            is_mortar = is_mortar_u or is_mortar_v
            
            if is_mortar:
                # 砂浆区域 - 创建凹陷效果
                # 计算到砂浆中心的距离
                center_u = 0.5
                center_v = 0.5
                
                if is_mortar_u:
                    dist_u = min(brick_u, 1.0 - brick_u) / mortar_threshold
                else:
                    dist_u = 1.0
                    
                if is_mortar_v:
                    dist_v = min(brick_v, 1.0 - brick_v) / mortar_threshold_v
                else:
                    dist_v = 1.0
                
                # 使用距离创建平滑的凹陷
                mortar_factor = min(dist_u, dist_v)
                depth = mortar_depth * (1.0 - smooth_step(0.0, 1.0, mortar_factor))
                
                # 砂浆的法线主要指向上方，有轻微的随机扰动
                normals[y, x, 0] = (math.sin(u * 50) * math.cos(v * 50)) * 0.1
                normals[y, x, 1] = (math.cos(u * 50) * math.sin(v * 50)) * 0.1
                normals[y, x, 2] = 0.8 + depth * 0.2
                
            else:
                # 砖块区域 - 创建砖块表面纹理
                # 基础深度
                base_depth = brick_depth
                
                # 添加砖块表面的细微纹理
                # 使用多个频率的噪声模拟砖块的粗糙表面
                surface_noise = 0.0
                
                # 低频大尺度变化
                surface_noise += math.sin(brick_u * math.pi * 2) * math.cos(brick_v * math.pi * 2) * 0.3
                
                # 中频纹理
                surface_noise += math.sin(brick_u * math.pi * 8) * math.cos(brick_v * math.pi * 6) * 0.2
                
                # 高频细节
                surface_noise += math.sin(brick_u * math.pi * 20) * math.cos(brick_v * math.pi * 16) * 0.1
                
                # 添加随机细节
                random_factor = math.sin(brick_u * 31.7 + brick_v * 17.3) * math.cos(brick_u * 23.1 + brick_v * 41.9)
                surface_noise += random_factor * 0.15
                
                # 计算砖块边缘的倒角效果
                edge_u = min(brick_u, 1.0 - brick_u) * 4.0  # 边缘距离
                edge_v = min(brick_v, 1.0 - brick_v) * 8.0  # 边缘距离
                edge_factor = min(edge_u, edge_v)
                edge_factor = smooth_step(0.0, 1.0, min(1.0, edge_factor))
                
                # 组合深度
                total_depth = base_depth + surface_noise * brick_roughness * edge_factor
                
                # 计算法线
                # X方向梯度
                dx = math.cos(brick_u * math.pi * 8) * math.pi * 8 * 0.2 / width
                dx += math.cos(brick_u * math.pi * 20) * math.pi * 20 * 0.1 / width
                
                # Y方向梯度  
                dy = -math.sin(brick_v * math.pi * 6) * math.pi * 6 * 0.2 / height
                dy += -math.sin(brick_v * math.pi * 16) * math.pi * 16 * 0.1 / height
                
                # 应用边缘因子
                dx *= edge_factor
                dy *= edge_factor
                
                normals[y, x, 0] = dx * 0.5
                normals[y, x, 1] = dy * 0.5
                normals[y, x, 2] = total_depth
    
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
    
    print(f"生成砖墙样式法线贴图 {width}x{height}...")
    
    # 生成砖墙法线贴图数据
    rgb_data = generate_brick_normal_map(width, height)
    
    # 保存为BMP文件
    output_file = "assets/brick_normal.bmp"
    save_bmp(output_file, rgb_data)
    
    print(f"砖墙法线贴图已保存到: {output_file}")
    print("砖墙法线贴图特征:")
    print("- 真实的砖块排列模式")
    print("- 砂浆凹陷效果")
    print("- 砖块表面纹理细节")
    print("- 砖块边缘倒角效果")
    print("- 适用于立方体各个面")
    print("\n使用方法:")
    print("1. 在程序中加载 cube1.obj")
    print("2. 加载 brick_normal.bmp 作为法线贴图")
    print("3. 启用法线贴图功能")
    print("4. 观察立方体的砖墙效果")

if __name__ == "__main__":
    main() 