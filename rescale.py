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


def extract_and_resize_cell(image, polygon, scale_factor, cell_size, neighborhood_size):
    """
    提取并缩放细胞图像及其邻域图像
    参数:
        image: 原始图像
        polygon: 多边形坐标
        scale_factor: 邻域扩展倍数
        cell_size: 细胞图像目标尺寸
        neighborhood_size: 邻域图像目标尺寸
    返回:
        cell_image: 缩放后的细胞图像
        neighborhood_image: 缩放后的邻域图像
    """
    # 获取多边形的边界框
    min_x, min_y, max_x, max_y = polygon.bounds
    
    # 检查多边形是否有效（面积非零）
    if polygon.is_empty or polygon.area <= 0:
        raise ValueError("无效的多边形: 面积为零或为空")
    
    # 检查边界框是否有效
    cell_width = max_x - min_x
    cell_height = max_y - min_y
    
    if cell_width <= 0 or cell_height <= 0:
        raise ValueError(f"无效的多边形边界框: 宽度={cell_width}, 高度={cell_height}")
    
    # 确保边界框完全在图像内
    min_x = max(0, int(min_x))
    min_y = max(0, int(min_y))
    max_x = min(image.shape[1]-1, int(max_x))
    max_y = min(image.shape[0]-1, int(max_y))
    
    # 再次检查边界框大小是否合法
    if min_x >= max_x or min_y >= max_y:
        # 如果边界框无效，创建一个简单的填充图像而不是抛出错误
        print(f"警告: 无效的多边形边界 [{min_x}, {min_y}, {max_x}, {max_y}], 使用填充图像代替")
        cell_image = np.zeros((cell_size, cell_size, 3), dtype=np.uint8)
        neighborhood_image = np.zeros((neighborhood_size, neighborhood_size, 3), dtype=np.uint8)
        return cell_image, neighborhood_image
    
    # 扩展邻域边界
    neighborhood_min_x = max(0, int(min_x - cell_width * scale_factor))
    neighborhood_min_y = max(0, int(min_y - cell_height * scale_factor))
    neighborhood_max_x = min(image.shape[1]-1, int(max_x + cell_width * scale_factor))
    neighborhood_max_y = min(image.shape[0]-1, int(max_y + cell_height * scale_factor))
    
    # 确保邻域边界框也是有效的
    if neighborhood_min_x >= neighborhood_max_x or neighborhood_min_y >= neighborhood_max_y:
        print(f"警告: 无效的邻域边界 [{neighborhood_min_x}, {neighborhood_min_y}, {neighborhood_max_x}, {neighborhood_max_y}], 使用填充图像代替")
        cell_image = np.zeros((cell_size, cell_size, 3), dtype=np.uint8)
        neighborhood_image = np.zeros((neighborhood_size, neighborhood_size, 3), dtype=np.uint8)
        return cell_image, neighborhood_image
    
    # 提取细胞区域和邻域区域
    cell_image = image[min_y:max_y, min_x:max_x]
    neighborhood_image = image[neighborhood_min_y:neighborhood_max_y, neighborhood_min_x:neighborhood_max_x]
    
    # 确保提取的图像不为空
    if cell_image.size == 0 or neighborhood_image.size == 0:
        print(f"警告: 提取的图像为空，使用填充图像代替")
        cell_image = np.zeros((cell_size, cell_size, 3), dtype=np.uint8)
        neighborhood_image = np.zeros((neighborhood_size, neighborhood_size, 3), dtype=np.uint8)
        return cell_image, neighborhood_image
    
    # 安全地缩放图像，处理特殊情况
    try:
        # 检查图像是否至少有1个像素
        if cell_image.shape[0] > 0 and cell_image.shape[1] > 0:
            resized_cell_image = resize(cell_image, (cell_size, cell_size), preserve_range=True).astype(np.uint8)
        else:
            resized_cell_image = np.zeros((cell_size, cell_size, 3), dtype=np.uint8)
            
        if neighborhood_image.shape[0] > 0 and neighborhood_image.shape[1] > 0:
            resized_neighborhood_image = resize(neighborhood_image, (neighborhood_size, neighborhood_size), preserve_range=True).astype(np.uint8)
        else:
            resized_neighborhood_image = np.zeros((neighborhood_size, neighborhood_size, 3), dtype=np.uint8)
    except Exception as e:
        print(f"缩放图像时出错: {e}, 使用填充图像代替")
        resized_cell_image = np.zeros((cell_size, cell_size, 3), dtype=np.uint8)
        resized_neighborhood_image = np.zeros((neighborhood_size, neighborhood_size, 3), dtype=np.uint8)
    
    return resized_cell_image, resized_neighborhood_image


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
            points.append((x, y))
    return Polygon(points)


def process_cells(image_path, cells_data_path, output_dir_cell, output_dir_neighborhood, start_cell_id=None):
    """
    处理所有细胞图像
    参数:
        image_path: 原始图像路径
        cells_data_path: 细胞数据CSV文件路径
        output_dir_cell: 细胞图像输出目录
        output_dir_neighborhood: 邻域图像输出目录
        start_cell_id: 可选，从指定的细胞ID开始处理，用于恢复中断的处理
    """
    # 创建输出目录
    os.makedirs(output_dir_cell, exist_ok=True)
    os.makedirs(output_dir_neighborhood, exist_ok=True)
    
    # 读取原始图像
    image = imread(image_path)
    
    # 读取细胞数据
    cells_data = pd.read_csv(cells_data_path, skiprows=1, names=['geometry', 'id'])
    
    # 如果指定了start_cell_id，找到对应的索引位置
    start_processing = False if start_cell_id else True
    
    # 处理每个细胞
    processed_count = 0
    error_count = 0
    
    for _, row in cells_data.iterrows():
        cell_id = row['id']
        
        # 检查是否应该开始处理
        if not start_processing:
            if cell_id == start_cell_id:
                start_processing = True
                print(f"从细胞ID {start_cell_id} 开始处理...")
            else:
                continue
        
        try:
            # 解析多边形坐标
            polygon = parse_polygon(row['geometry'])
            
            # 提取并缩放细胞图像及其邻域图像
            cell_image, neighborhood_image = extract_and_resize_cell(image, polygon, scale_factor=16, cell_size=16, neighborhood_size=256)
            
             # 对细胞图像和邻域图像进行垂直翻转和顺时针旋转90度
            #cell_image = np.flipud(cell_image)
            #cell_image = np.fliplr(cell_image)
            cell_image = rotate(cell_image, -90, resize=True, mode='reflect', preserve_range=True).astype(np.uint8)  # 顺时针旋转90度
            #neighborhood_image = np.flipud(neighborhood_image)  # 垂直翻转
            #neighborhood_image = np.fliplr(neighborhood_image)
            neighborhood_image = rotate(neighborhood_image, -90, resize=True, mode='reflect', preserve_range=True).astype(np.uint8)  # 顺时针旋转90度
        
            # 保存图像
            imsave(os.path.join(output_dir_cell, f"{cell_id}_16.tif"), cell_image)
            imsave(os.path.join(output_dir_neighborhood, f"{cell_id}_256.tif"), neighborhood_image)
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


def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True,
                      help='输入图像路径')
    parser.add_argument('--cells', type=str, required=True,
                      help='细胞数据CSV文件路径')
    parser.add_argument('--output-dir', type=str, required=True,
                      help='输出目录')
    parser.add_argument('--start-from', type=str, default=None,
                      help='从指定的细胞ID开始处理，用于恢复中断的处理')
    return parser.parse_args()


def main():
    """主函数，协调整个处理流程"""
    args = get_args()
    
    # 创建细胞图像和邻域图像的输出目录
    output_dir_cell = os.path.join(args.output_dir, 'cells_16')
    output_dir_neighborhood = os.path.join(args.output_dir, 'cells_256')
    
    # 处理细胞图像及其邻域图像
    process_cells(args.image, args.cells, output_dir_cell, output_dir_neighborhood, args.start_from)


if __name__ == '__main__':
    # 当脚本直接运行时执行main函数
    main()