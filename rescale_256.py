import argparse  # 用于解析命令行参数
import os  # 用于文件和目录操作
import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from skimage.io import imread, imsave
from skimage.transform import resize, rotate
import re  # 用于正则表达式解析

# 从自定义utils模块导入工具函数
from utils import (
        load_image, save_image, read_string, write_string,
        load_tsv, save_tsv)

pixel_size_raw = float(0.27371559375573706)
#pixel_size_raw = float(0.5)
pixel_size = float(0.5)
scale = pixel_size_raw / pixel_size

def get_image_filename(prefix):
    """
    根据前缀查找存在的图像文件
    参数:
        prefix: 文件路径前缀
    返回:
        找到的图像文件名
    异常:
        FileNotFoundError: 当找不到图像文件时抛出
    """
    file_exists = False
    # 尝试常见图像后缀
    for suffix in ['.jpg', '.png', '.tiff']:
        filename = prefix + suffix
        if os.path.exists(filename):
            file_exists = True
            break
    if not file_exists:
        raise FileNotFoundError('Image not found')
    return filename


def parse_polygon(polygon_str):
    """
    解析多边形字符串
    参数:
        polygon_str: 多边形字符串，格式为 "POLYGON ((x1 y1, x2 y2, ..., xn yn))"
    返回:
        shapely.geometry.Polygon 对象
    """
    # 使用正则表达式提取坐标
    # 匹配 POLYGON ((...)) 中的内容
    match = re.search(r'POLYGON\s*\(\s*\(\s*(.*?)\s*\)\s*\)', polygon_str)
    if not match:
        raise ValueError("Invalid polygon format")
    
    coords = match.group(1)
    points = []
    for point in coords.split(','):
        point = point.strip()
        if point:
            # 去掉括号并分割 x 和 y 坐标
            point = point.replace('(', '').replace(')', '')
            x, y = map(float, point.split())
            # 缩放坐标
            x_scaled = x * scale
            y_scaled = y * scale
            points.append((x_scaled, y_scaled))
    return Polygon(points)


def extract_256_image(image, polygon):
    """
    根据多边形中心提取 256*256 的细胞图像，超出边界用 0 填充
    参数:
        image: 原始图像
        polygon: 多边形坐标
    返回:
        256*256 的细胞图像
    """
    # 获取多边形的中心
    center_x, center_y = polygon.centroid.x, polygon.centroid.y
    
    # 计算提取区域的边界
    half_size = 128
    start_x = int(center_x - half_size)
    start_y = int(center_y - half_size)
    end_x = start_x + 256
    end_y = start_y + 256
    
    # 创建一个全零的 256*256 图像
    extracted_image = np.zeros((256, 256, image.shape[2]), dtype=image.dtype)
    
    # 计算在原始图像中的有效区域
    valid_start_x = max(0, start_x)
    valid_start_y = max(0, start_y)
    valid_end_x = min(image.shape[1], end_x)
    valid_end_y = min(image.shape[0], end_y)
    
    # 计算在提取图像中的对应区域
    extract_start_x = valid_start_x - start_x
    extract_start_y = valid_start_y - start_y
    extract_end_x = extract_start_x + (valid_end_x - valid_start_x)
    extract_end_y = extract_start_y + (valid_end_y - valid_start_y)
    
    # 复制有效区域到提取图像中
    extracted_image[extract_start_y:extract_end_y, extract_start_x:extract_end_x] = image[valid_start_y:valid_end_y, valid_start_x:valid_end_x]
    
    return extracted_image


def process_cells(image_path, cells_data_path, output_dir):
    """
    处理所有细胞图像
    参数:
        image_path: 原始图像路径
        cells_data_path: 细胞数据CSV文件路径
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取原始图像
    image = imread(image_path)
    
    # 读取细胞数据
    cells_data = pd.read_csv(cells_data_path, skiprows=1, names=['geometry', 'id'])
    
    # 处理每个细胞
    processed_count = 0
    error_count = 0
    
    for _, row in cells_data.iterrows():
        cell_id = row['id']
        
        try:
            # 解析多边形坐标
            polygon = parse_polygon(row['geometry'])
            
            # 提取 256*256 的细胞图像
            cell_image = extract_256_image(image, polygon)
            #cell_image = np.flipud(cell_image)
            #cell_image = np.fliplr(cell_image)
            cell_image = rotate(cell_image, -90, resize=True, mode='reflect', preserve_range=True).astype(np.uint8)  # 顺时针旋转90度
            # 保存图像
            imsave(os.path.join(output_dir, f"{cell_id}_256.tif"), cell_image)
            processed_count += 1
            
            # 每处理100个细胞输出一次进度
            if processed_count % 100 == 0:
                print(f"已处理 {processed_count} 个细胞，当前ID: {cell_id}...")
        except Exception as e:
            print(f"处理细胞 {cell_id} 时出错: {e}")
            error_count += 1
            # 如果错误太多，退出处理
            if error_count > 100:
                print("错误数量过多，停止处理")
                break
    
    print(f"处理完成，共处理 {processed_count} 个细胞，遇到 {error_count} 个错误")


def test_image_transformation(input_path, output_path):
    """
    测试图像翻转和旋转功能
    """
    image = imread(input_path)
    transformed_image = np.fliplr(image)
    transformed_image = rotate(transformed_image, 90, resize=True, mode='reflect', preserve_range=True).astype(np.uint8)
    imsave(output_path, transformed_image)

def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='输入图像路径')
    parser.add_argument('--cells', type=str, help='细胞数据CSV文件路径')
    parser.add_argument('--output-dir', type=str, help='输出目录')
    parser.add_argument('--test-image', type=str, help='测试图像路径')
    return parser.parse_args()


def main():
    """主函数，协调整个处理流程"""
    args = get_args()
    
    if args.test_image:
        test_image_transformation(args.test_image, "output.png")
        print("测试图像已保存为 output.png")
    elif args.image and args.cells and args.output_dir:
        # 处理细胞图像
        process_cells(args.image, args.cells, args.output_dir)
    else:
        print("请提供必要的参数，使用 --test-image 进行测试，或提供 --image、--cells 和 --output-dir 进行细胞图像处理。")


if __name__ == '__main__':
    # 当脚本直接运行时执行main函数
    main()