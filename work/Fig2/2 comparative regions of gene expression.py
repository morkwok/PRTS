

#导入模块
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import anndata
import geopandas as gpd
import scanpy as sc
import tensorflow as tf

from tifffile import imread, imwrite
from csbdeep.utils import normalize
from stardist.models import StarDist2D
from shapely.geometry import Polygon, Point
from scipy import sparse
from matplotlib.colors import ListedColormap
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'


dir_base = 'E:/SCI/002/data/brainfixfrozen/Visium_HD_Mouse_Brain_Fixed_Frozen_binned_outputs/binned_outputs/square_002um/'
#图像的文件名
filename = 'Visium_HD_Mouse_Brain_Fixed_Frozen_tissue_image.btf'
img = imread(dir_base + filename)

#细胞坐标信息

import geopandas as gpd
gdf = gpd.read_file("E:/pythonproject/project002-2/area-Frozen.shp")



gdf.head()


# Define a single color cmap 定义一个单色cmap
cmap=ListedColormap(['grey'])

# Load Visium HD data 加载Visium HD数据
raw_h5_file = 'E:/pythonproject/project002-2/BrainHD_results-Frozen.h5ad'
count_area_filtered_adata = sc.read_h5ad(raw_h5_file)

dir_base = 'E:/SCI/002/picture/Fig2/'


import matplotlib.pyplot as plt
import numpy as np  # 用于颜色生成

def plot_bboxes_and_save_image(title, img, bboxes=None, colors=None, crop_bbox=None, output_name=None):

    # 图像预处理
    if crop_bbox:
        x1, y1, x2, y2 = crop_bbox
        base_img = img[y1:y2, x1:x2]
    else:
        base_img = img
        x1 = y1 = 0

    # 创建画布
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(base_img, cmap='gray', origin='lower')
    ax.set_title(title)
    ax.axis('off')

    # 自动生成颜色（如果未提供）
    if bboxes:
        n_boxes = len(bboxes)
        if not colors:  # 使用hsv色环生成鲜明颜色
            colors = plt.cm.hsv(np.linspace(0, 1, n_boxes))
        elif len(colors) < n_boxes:  # 颜色不足时循环使用
            colors = [colors[i % len(colors)] for i in range(n_boxes)]

        # 绘制所有bbox
        for i, bbox in enumerate(bboxes):
            # 坐标转换
            bx1, by1, bx2, by2 = bbox
            if crop_bbox:
                bx1 -= x1
                by1 -= y1
                bx2 -= x1
                by2 -= y1

            # 创建彩色矩形框
            rect = plt.Rectangle(
                (bx1, by1),
                bx2 - bx1,
                by2 - by1,
                linewidth=1.5,  # 加粗线宽
                edgecolor=colors[i],  # 按索引取对应颜色
                facecolor='none',
                linestyle='--'  # 可选虚线样式
            )
            ax.add_patch(rect)

    # 输出控制
    if output_name:
        plt.savefig(output_name, bbox_inches='tight', dpi=300)  # 提高分辨率
        plt.close()
    else:
        plt.show()

# 使用示例 - 自定义颜色方案
custom_colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00']  # 红、绿、蓝、品红



# 定义多个bbox区域
bbox_list = [
    [8000,15000,13000,20000],  # 原始坐标系下的区域1
    [5000,6000,10000,11000],   # 原始坐标系下的区域2
    [3000,3000,8000,8000],
    [6000,8000,11000,13000]
]

# 定义裁剪区域
#crop_area = [50, 150, 750, 850]

plot_bboxes_and_save_image(
    title="comparative regions of gene expression",
    img=img,
    bboxes=bbox_list,
    colors=custom_colors,  # 传入自定义颜色
    crop_bbox=None,
    output_name=dir_base+"F2c.png"
)



