#!/usr/bin/env python3
import struct
import os

def create_bmp_texture(filename, width, height):
    """创建一个简单的BMP纹理文件"""
    
    # 确保assets目录存在
    os.makedirs('assets', exist_ok=True)
    
    # 计算行对齐（每行必须是4字节的倍数）
    row_size = ((width * 3 + 3) // 4) * 4
    image_size = row_size * height
    file_size = 54 + image_size
    
    with open(filename, 'wb') as f:
        # BMP文件头 (14字节)
        f.write(b'BM')                          # 签名
        f.write(struct.pack('<I', file_size))   # 文件大小
        f.write(struct.pack('<HH', 0, 0))       # 保留字段
        f.write(struct.pack('<I', 54))          # 数据偏移
        
        # DIB信息头 (40字节)
        f.write(struct.pack('<I', 40))          # 信息头大小
        f.write(struct.pack('<I', width))       # 图像宽度
        f.write(struct.pack('<I', height))      # 图像高度
        f.write(struct.pack('<H', 1))           # 颜色平面数
        f.write(struct.pack('<H', 24))          # 每像素位数
        f.write(struct.pack('<I', 0))           # 压缩类型
        f.write(struct.pack('<I', image_size))  # 图像大小
        f.write(struct.pack('<I', 2835))        # X分辨率 (72 DPI)
        f.write(struct.pack('<I', 2835))        # Y分辨率 (72 DPI)  
        f.write(struct.pack('<I', 0))           # 颜色表大小
        f.write(struct.pack('<I', 0))           # 重要颜色数
        
        # 写入像素数据（BMP从底部开始存储）
        for y in range(height - 1, -1, -1):
            row_data = bytearray()
            for x in range(width):
                # 创建棋盘格纹理
                check_size = 32
                is_white = ((x // check_size) + (y // check_size)) % 2 == 0
                
                if is_white:
                    r, g, b = 255, 255, 255  # 白色
                else:
                    r, g, b = 255, 0, 0      # 红色
                
                # BMP存储顺序为BGR
                row_data.extend([b, g, r])
            
            # 添加行对齐填充
            padding = row_size - (width * 3)
            row_data.extend([0] * padding)
            
            f.write(row_data)
    
    print(f"创建了纹理文件: {filename} ({width}x{height})")

def create_gradient_texture(filename, width, height):
    """创建一个渐变纹理"""
    
    # 确保assets目录存在
    os.makedirs('assets', exist_ok=True)
    
    # 计算行对齐
    row_size = ((width * 3 + 3) // 4) * 4
    image_size = row_size * height
    file_size = 54 + image_size
    
    with open(filename, 'wb') as f:
        # BMP文件头 (14字节)
        f.write(b'BM')
        f.write(struct.pack('<I', file_size))
        f.write(struct.pack('<HH', 0, 0))
        f.write(struct.pack('<I', 54))
        
        # DIB信息头 (40字节)
        f.write(struct.pack('<I', 40))
        f.write(struct.pack('<I', width))
        f.write(struct.pack('<I', height))
        f.write(struct.pack('<H', 1))
        f.write(struct.pack('<H', 24))
        f.write(struct.pack('<I', 0))
        f.write(struct.pack('<I', image_size))
        f.write(struct.pack('<I', 2835))
        f.write(struct.pack('<I', 2835))
        f.write(struct.pack('<I', 0))
        f.write(struct.pack('<I', 0))
        
        # 写入像素数据
        for y in range(height - 1, -1, -1):
            row_data = bytearray()
            for x in range(width):
                # 创建彩色渐变
                r = int((x / width) * 255)
                g = int((y / height) * 255)
                b = int(((x + y) / (width + height)) * 255)
                
                # BMP存储顺序为BGR
                row_data.extend([b, g, r])
            
            # 添加行对齐填充
            padding = row_size - (width * 3)
            row_data.extend([0] * padding)
            
            f.write(row_data)
    
    print(f"创建了渐变纹理文件: {filename} ({width}x{height})")

if __name__ == "__main__":
    # 创建不同的测试纹理
    create_bmp_texture("assets/texture.bmp", 256, 256)
    create_bmp_texture("assets/cube_texture.bmp", 128, 128)
    create_gradient_texture("assets/gradient.bmp", 512, 512)
    
    print("所有纹理文件创建完成！") 