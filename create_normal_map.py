#!/usr/bin/env python3
"""
生成法线贴图的脚本
为正方体创建表面凹凸不平的法线贴图
输出文件: normal1.bmp
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

def generate_bumpy_normal_map(width, height):
    """
    生成凹凸不平的法线贴图
    使用多个噪声函数叠加创建复杂的表面细节
    """
    # 创建法线数组 (范围 -1 到 1)
    normals = np.zeros((height, width, 3), dtype=np.float32)
    
    # 基础法线 (指向Z正方向)
    normals[:, :, 2] = 1.0
    
    # 生成多层噪声
    for octave in range(5):
        frequency = 2 ** octave * 4  # 频率递增
        amplitude = 0.5 ** octave    # 振幅递减
        
        # 生成噪声高度图
        for y in range(height):
            for x in range(width):
                # 标准化坐标到 [0, 1]
                u = x / width
                v = y / height
                
                # 多个正弦波叠加创建复杂图案
                noise = (
                    math.sin(u * frequency * math.pi) * 
                    math.cos(v * frequency * math.pi) +
                    math.sin(u * frequency * 2 * math.pi + 1.5) * 
                    math.cos(v * frequency * 1.5 * math.pi + 0.7) +
                    math.sin(u * frequency * 0.5 * math.pi + 2.1) * 
                    math.cos(v * frequency * 3 * math.pi + 1.2)
                ) / 3.0
                
                # 添加到法线的X和Y分量
                normals[y, x, 0] += noise * amplitude * 0.8  # X方向扰动
                normals[y, x, 1] += noise * amplitude * 0.6  # Y方向扰动
    
    # 添加细节噪声
    for y in range(height):
        for x in range(width):
            u = x / width
            v = y / height
            
            # 高频细节
            detail = (
                math.sin(u * 32 * math.pi) * math.cos(v * 32 * math.pi) * 0.1 +
                math.sin(u * 64 * math.pi + 1.0) * math.cos(v * 48 * math.pi + 0.5) * 0.05
            )
            
            normals[y, x, 0] += detail
            normals[y, x, 1] += detail * 0.8
    
    # 归一化法线向量
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
    
    print(f"生成 {width}x{height} 法线贴图...")
    
    # 生成法线贴图数据
    rgb_data = generate_bumpy_normal_map(width, height)
    
    # 保存为BMP文件
    output_file = "assets/normal1.bmp"
    save_bmp(output_file, rgb_data)
    
    print(f"法线贴图已保存到: {output_file}")
    print("法线贴图特征:")
    print("- 多层噪声叠加")
    print("- 复杂的凹凸表面细节")
    print("- 适用于正方体表面")
    print("- RGB格式: R=X法线, G=Y法线, B=Z法线")

if __name__ == "__main__":
    main() 