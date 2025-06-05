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




#定义函数
# General image plotting functions 一般图像绘图函数


def plot_gene_and_save_image(title, gdf, gene, img, adata, bbox=None, output_name=None):

    if bbox is not None:
        # Crop the image to the bounding box
        cropped_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    else:
        cropped_img = img

    # Plot options
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the cropped image
    axes[0].imshow(cropped_img, cmap='gray', origin='lower')
    axes[0].set_title(title)
    axes[0].axis('off')

    # Create filtering polygon
    if bbox is not None:
        bbox_polygon = Polygon([(bbox[0], bbox[1]), (bbox[2], bbox[1]), (bbox[2], bbox[3]), (bbox[0], bbox[3])])


    # Find a gene of interest and merge with the geodataframe
    gene_expression = adata[:, gene].to_df()
    gene_expression['id'] = gene_expression.index
    merged_gdf = gdf.merge(gene_expression, left_on='id', right_on='id')

    if bbox is not None:
        # Filter for polygons in the box
        intersects_bbox = merged_gdf['geometry'].intersects(bbox_polygon)
        filtered_gdf = merged_gdf[intersects_bbox]
    else:
        filtered_gdf = merged_gdf

    # Plot the filtered polygons on the second axis
    filtered_gdf.plot(column=gene, cmap='inferno', legend=True, ax=axes[1])
    axes[1].set_title(gene)
    axes[1].axis('off')
    axes[1].legend(loc='upper left', bbox_to_anchor=(1.05, 1))

    # Save the plot if output_name is provided
    if output_name is not None:
        plt.savefig(output_name, bbox_inches='tight')  # Use bbox_inches='tight' to include the legend
    else:
        plt.show()


sc.settings.set_figure_params(dpi=50, dpi_save=300, figsize=(7, 7))


dir_base = 'E:/SCI/002/data/BA3/'
#图像的文件名
filename = 'Visium_FFPE_Mouse_Brain_image.tif'
img = imread(dir_base + filename)

#细胞坐标信息

import geopandas as gpd
gdf = gpd.read_file("E:/pythonproject/project002-2/area-BA3.shp")



gdf.head()


# Define a single color cmap 定义一个单色cmap
cmap=ListedColormap(['grey'])

# Load Visium HD data 加载Visium HD数据
raw_h5_file = 'E:/SCI/002/data/prediction/ba3-prediction.h5ad'
count_area_filtered_adata = sc.read_h5ad(raw_h5_file)

dir_base = 'E:/SCI/002/picture/S10/'

# Calculate quality control metrics for the filtered AnnData object
# 计算过滤后的AnnData对象的质量控制度量
sc.pp.calculate_qc_metrics(count_area_filtered_adata, inplace=True)

# Normalize total counts for each cell in the AnnData object
# 规范化AnnData对象中每个单元格的总数
sc.pp.normalize_total(count_area_filtered_adata, inplace=True)

# Logarithmize the values in the AnnData object after normalization
# 在归一化之后对AnnData对象中的值取对数
sc.pp.log1p(count_area_filtered_adata)

# Identify highly variable genes in the dataset using the Seurat method
# 使用Seurat方法在数据集中识别高度可变的基因
sc.pp.highly_variable_genes(count_area_filtered_adata, flavor="seurat", n_top_genes=600)

# Perform Principal Component Analysis (PCA) on the AnnData object
# 对AnnData对象执行主成分分析（PCA）
sc.pp.pca(count_area_filtered_adata)


#
sc.pl.pca_variance_ratio(count_area_filtered_adata, log=True)



# Build a neighborhood graph based on PCA components
# 构建基于PCA分量的邻域图
sc.pp.neighbors(count_area_filtered_adata)

# Perform Leiden clustering on the neighborhood graph and store the results in 'clusters' column
# 对邻域图执行Leiden聚类，并将结果存储在“clusters”列中
# Adjust the resolution parameter as needed for different samples
# 根据需要调整不同样品的分辨率参数



sc.tl.umap(count_area_filtered_adata)

#sc.pl.umap(count_area_filtered_adata)

sc.tl.leiden(count_area_filtered_adata, resolution=0.6, key_added="clusters1")

#sc.pl.umap(count_area_filtered_adata, color='clusters1')


#定义函数
# General image plotting functions 一般图像绘图函数
def plot_mask_and_save_image(title, gdf, img, cmap, output_name=None, bbox=None):
    if bbox is not None:
        # Crop the image to the bounding box
        cropped_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    else:
        cropped_img = img

    # Plot options
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the cropped image
    axes[0].imshow(cropped_img, cmap='gray', origin='lower')
    axes[0].set_title(title)
    axes[0].axis('off')

    # Create filtering polygon
    if bbox is not None:
        bbox_polygon = Polygon([(bbox[0], bbox[1]), (bbox[2], bbox[1]), (bbox[2], bbox[3]), (bbox[0], bbox[3])])
        # Filter for polygons in the box
        intersects_bbox = gdf['geometry'].intersects(bbox_polygon)
        filtered_gdf = gdf[intersects_bbox]
    else:
        filtered_gdf=gdf

    # Plot the filtered polygons on the second axis
    filtered_gdf.plot(cmap=cmap, ax=axes[1])
    axes[1].axis('off')
    axes[1].legend(loc='upper left', bbox_to_anchor=(1.05, 1))


    # Save the plot if output_name is provided
    if output_name is not None:
        plt.savefig(output_name, bbox_inches='tight')  # Use bbox_inches='tight' to include the legend
    else:
        plt.show()

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

# Calculate quality control metrics for the filtered AnnData object
# 计算过滤后的AnnData对象的质量控制度量
sc.pp.calculate_qc_metrics(count_area_filtered_adata, inplace=True)

print("Matrix obs:",count_area_filtered_adata.obs)

plot_qc_metrics_and_save(
    title="Hippocampus Slice",
    gdf=gdf,
    adata=count_area_filtered_adata,
    img=img,
   # bbox=(x0, y0, x1, y1),  # 可选区域
    output_name=dir_base+"qc_metrics-BA3-预测.png"
)


#marker系列-Fig2b
#神经元
#'''''''''
#plot_gene_and_save_image(title="Lateral habenula", gdf=gdf,bbox=(8000,15000,13000,20000), gene='Kcnma1', img=img, adata=count_area_filtered_adata,output_name=dir_base+"Fig2b-1.pre.tiff")
plot_gene_and_save_image(title="Complete histological image",gdf=gdf, gene='Kcnma1', img=img, adata=count_area_filtered_adata,output_name=dir_base+"S10-1.pre.tiff")
#'''''''''


#少突胶质
#plot_gene_and_save_image(title="cortex 1", gdf=gdf,bbox=(5000,6000,10000,11000), gene='Plp1', img=img, adata=count_area_filtered_adata,output_name=dir_base+"Fig2b-2.pre.tiff")
plot_gene_and_save_image(title="Complete histological image",gdf=gdf, gene='Plp1', img=img, adata=count_area_filtered_adata,output_name=dir_base+"S10-2.pre.tiff")


#星形胶质
#plot_gene_and_save_image(title="Meninges", gdf=gdf,bbox=(3000,3000,8000,8000), gene='Ptgds', img=img, adata=count_area_filtered_adata,output_name=dir_base+"Fig2b-3.pre.tiff")
plot_gene_and_save_image(title="Complete histological image",gdf=gdf, gene='Ptgds', img=img, adata=count_area_filtered_adata,output_name=dir_base+"S10-3.pre.tiff")


#脉络丛上皮细胞
#plot_gene_and_save_image(title="Hippocampus and lateral ventricles", gdf=gdf,bbox=(6000,8000,11000,13000), gene='Ttr', img=img, adata=count_area_filtered_adata,output_name=dir_base+"Fig2b-4.pre.tiff")
plot_gene_and_save_image(title="Complete histological image",gdf=gdf, gene='Ttr', img=img, adata=count_area_filtered_adata,output_name=dir_base+"S10-4.pre.tiff")


#多细胞marker且重要
#plot_gene_and_save_image(title="Hippocampus and lateral ventricles", gdf=gdf,bbox=(6000,8000,11000,13000), gene='Apoe', img=img, adata=count_area_filtered_adata,output_name=dir_base+"Fig2b-5.pre.tiff")
plot_gene_and_save_image(title="Complete histological image",gdf=gdf, gene='Apoe', img=img, adata=count_area_filtered_adata,output_name=dir_base+"S10-5.pre.tiff")

plot_mask_and_save_image(title="Region of Interest 2",gdf=gdf,cmap=cmap,img=img,output_name=dir_base+"BA3image_ALL-mask.tif")

plot_mask_and_save_image(title="Region of Interest 1",gdf=gdf,bbox=(13844,10136,17760,12333),cmap=cmap,img=img,output_name=dir_base+"BA3image_1-mask.ROI1.tif")
