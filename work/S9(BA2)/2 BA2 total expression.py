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
import anndata
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'


def plot_qc_metrics_and_save(title, gdf, adata, img, bbox=None, output_name=None):
    """绘制总counts和总features的空间分布"""

    # 图像裁剪逻辑保持不变
    if bbox is not None:
        cropped_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    else:
        cropped_img = img

    # 修改为 3 个子图布局（原图 + counts + features）
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # 宽度调整为18以容纳三个子图

    # ---- 原始切片图像 ----
    axes[0].imshow(cropped_img, cmap='gray', origin='lower')
    axes[0].set_title(f"{title} - Original")
    axes[0].axis('off')

    # ---- 数据准备 ----
    # 计算总counts和features
    #adata.obs['total_counts'] = adata.X.sum(axis=1).A1  # 总UMI counts
    adata.obs['n_features'] = adata.obs['n_genes_by_counts']  # 检测到的基因数

    # 合并到地理数据
    merged_gdf = gdf.merge(
        adata.obs[['total_counts', 'n_features']],
        left_on='id',
        right_index=True
    )

    # 空间过滤逻辑
    if bbox is not None:
        bbox_polygon = Polygon([(bbox[0], bbox[1]), (bbox[2], bbox[1]),
                                (bbox[2], bbox[3]), (bbox[0], bbox[3])])
        intersects_bbox = merged_gdf['geometry'].intersects(bbox_polygon)
        filtered_gdf = merged_gdf[intersects_bbox]
    else:
        filtered_gdf = merged_gdf

    # ---- 绘制总Counts ----
    vmax_counts = filtered_gdf['total_counts'].quantile(0.99)  # 动态设置颜色范围
    filtered_gdf.plot(
        column='total_counts',
        cmap='viridis',
        legend=True,
        ax=axes[1],
        vmax=vmax_counts,
        legend_kwds={'label': "Total UMI Counts"}
    )
    axes[1].set_title("Total UMI Counts")
    axes[1].axis('off')

    # ---- 绘制总Features ----
    vmax_features = filtered_gdf['n_features'].quantile(0.99)
    filtered_gdf.plot(
        column='n_features',
        cmap='plasma',
        legend=True,
        ax=axes[2],
        vmax=vmax_features,
        legend_kwds={'label': "Number of Features"}
    )
    axes[2].set_title("Detected Features")
    axes[2].axis('off')

    # 调整布局并保存/显示
    plt.tight_layout()
    if output_name:
        plt.savefig(output_name, bbox_inches='tight', dpi=300)
    else:
        plt.show()


dir_base = 'E:/SCI/002/data/BA2/'
#图像的文件名
filename = 'CytAssist_FFPE_Mouse_Brain_Rep2_tissue_image.tif'
img = imread(dir_base + filename)



#细胞坐标信息
gdf = gpd.read_file("E:/pythonproject/project002-2/area-BA2.shp")



gdf.head()



# Define a single color cmap 定义一个单色cmap
cmap=ListedColormap(['grey'])

# Load Visium HD data 加载Visium HD数据
raw_h5_file = 'E:/SCI/002/data/prediction/BA2-prediction.h5ad'
count_area_filtered_adata = sc.read_h5ad(raw_h5_file)

dir_base = 'E:/SCI/002/picture/S9/'

# Calculate quality control metrics for the filtered AnnData object
# 计算过滤后的AnnData对象的质量控制度量
sc.pp.calculate_qc_metrics(count_area_filtered_adata, inplace=True)

print("Matrix obs:",count_area_filtered_adata.obs)

plot_qc_metrics_and_save(
    title="Complete histological image",
    gdf=gdf,
    adata=count_area_filtered_adata,
    img=img,
   # bbox=(x0, y0, x1, y1),  # 可选区域
    output_name=dir_base+"S9c-预测.png"
)


