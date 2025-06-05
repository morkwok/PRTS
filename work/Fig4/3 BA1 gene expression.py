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

sc.settings.set_figure_params(dpi=50, dpi_save=300, figsize=(7, 7))


def plot_gene_overlay( title, gdf,  gene, img, adata, bbox=None,output_name=None,img_alpha=0.3,gene_alpha=0.7, figsize=(10, 10)):

    fig, ax = plt.subplots(figsize=figsize)

    # ========== 病理图片处理 ==========
    if bbox is not None:
        # 裁剪图片并设置坐标范围
        cropped_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        ax.imshow(
            cropped_img,
            cmap='gray',
            origin='lower',
            extent=[bbox[0], bbox[2], bbox[1], bbox[3]],  # (xmin, xmax, ymin, ymax)
            alpha=img_alpha
        )
    else:
        ax.imshow(img, cmap='gray', origin='lower', alpha=img_alpha)

    # ========== 基因数据处理 ==========
    # 提取目标基因表达数据
    gene_expression = adata[:, gene].to_df()
    gene_expression['id'] = gene_expression.index

    # 合并空间坐标与基因表达
    merged_gdf = gdf.merge(gene_expression, on='id')

    # 空间过滤
    if bbox is not None:
        bbox_polygon = Polygon([
            (bbox[0], bbox[1]),
            (bbox[2], bbox[1]),
            (bbox[2], bbox[3]),
            (bbox[0], bbox[3])
        ])
        filtered_gdf = merged_gdf[merged_gdf.intersects(bbox_polygon)]
    else:
        filtered_gdf = merged_gdf

    # ========== 基因热图绘制 ==========
    # 绘制半透明热图
    filtered_gdf.plot(
        column=gene,
        cmap='inferno',
        alpha=gene_alpha*1.2,
        legend=True,
        ax=ax,
        legend_kwds={
            'label': f"{gene} Expression Intensity",
            'shrink': 0.6,
            'location': 'right'
        }
    )

    # ========== 图像修饰 ==========
    # 设置坐标范围
    if bbox is not None:
        ax.set_xlim(bbox[0], bbox[2])
        ax.set_ylim(bbox[1], bbox[3])

    ax.set_title(f"{title}\nGene: {gene}", fontsize=14, pad=20)
    ax.axis('off')

    # 调整图例位置
    legend = ax.get_legend()
    if legend:
        legend.set_bbox_to_anchor((1.3, 0.7))  # 右侧外部

    # ========== 输出处理 ==========
    if output_name:
        plt.savefig(
            output_name,
            bbox_inches='tight',
            dpi=300,
            transparent=True  # 透明背景
        )
        print(f"Saved to: {output_name}")
    else:
        plt.show()

    plt.close()

def plot_all_area_gene_overlay( title, gdf,  gene, img, adata, bbox=None,output_name=None,img_alpha=0,gene_alpha=0.7, figsize=(10, 10)):



    fig, ax = plt.subplots(figsize=figsize)

    # ========== 病理图片处理 ==========
    if bbox is not None:
        # 裁剪图片并设置坐标范围
        cropped_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        ax.imshow(
            cropped_img,
            cmap='gray',
            origin='lower',
            extent=[bbox[0], bbox[2], bbox[1], bbox[3]],  # (xmin, xmax, ymin, ymax)
            alpha=img_alpha
        )
    else:
        ax.imshow(img, cmap='gray', origin='lower', alpha=img_alpha)

    # ========== 基因数据处理 ==========
    # 提取目标基因表达数据
    gene_expression = adata[:, gene].to_df()
    gene_expression['id'] = gene_expression.index

    # 合并空间坐标与基因表达
    merged_gdf = gdf.merge(gene_expression, on='id')

    # 空间过滤
    if bbox is not None:
        bbox_polygon = Polygon([
            (bbox[0], bbox[1]),
            (bbox[2], bbox[1]),
            (bbox[2], bbox[3]),
            (bbox[0], bbox[3])
        ])
        filtered_gdf = merged_gdf[merged_gdf.intersects(bbox_polygon)]
    else:
        filtered_gdf = merged_gdf

    # ========== 基因热图绘制 ==========
    # 绘制半透明热图
    filtered_gdf.plot(
        column=gene,
        cmap='inferno',
        alpha=gene_alpha*1.2,
        legend=True,
        ax=ax,
        legend_kwds={
            'label': f"{gene} Expression Intensity",
            'shrink': 0.6,
            'location': 'right'
        }
    )

    # ========== 图像修饰 ==========
    # 设置坐标范围
    if bbox is not None:
        ax.set_xlim(bbox[0], bbox[2])
        ax.set_ylim(bbox[1], bbox[3])

    ax.set_title(f"{title}\nGene: {gene}", fontsize=14, pad=20)
    ax.axis('off')

    # 调整图例位置
    legend = ax.get_legend()
    if legend:
        legend.set_bbox_to_anchor((1.3, 0.7))  # 右侧外部

    # ========== 输出处理 ==========
    if output_name:
        plt.savefig(
            output_name,
            bbox_inches='tight',
            dpi=300,
            transparent=True  # 透明背景
        )
        print(f"Saved to: {output_name}")
    else:
        plt.show()

    plt.close()

dir_base = 'E:/SCI/002/data/BA1/'
#图像的文件名
filename = 'CytAssist_FFPE_Mouse_Brain_Rep1_tissue_image.tif'
img = imread(dir_base + filename)


#细胞坐标信息

import geopandas as gpd
gdf = gpd.read_file("E:/pythonproject/project002-2/area-BA1.shp")



gdf.head()


# Load Visium HD data 加载Visium HD数据
raw_h5_file = 'E:/SCI/002/data/prediction/BA1-prediction.h5ad'
count_area_filtered_adata = sc.read_h5ad(raw_h5_file)

dir_base = 'E:/SCI/002/picture/Fig4/'

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
sc.tl.leiden(count_area_filtered_adata, resolution=0.6, key_added="clusters1")




#marker系列-Fig2b
#神经元
#'''''''''
plot_all_area_gene_overlay(title="Complete histological image-prediction",gdf=gdf, gene='Kcnma1', img=img, adata=count_area_filtered_adata,output_name=dir_base+"F4e-1.pre.tiff")

#少突胶质
plot_all_area_gene_overlay(title="Complete histological image-prediction",gdf=gdf, gene='Plp1', img=img, adata=count_area_filtered_adata,output_name=dir_base+"F4e-2.pre.tiff")

#星形胶质
plot_all_area_gene_overlay(title="Complete histological image-prediction",gdf=gdf, gene='Ptgds', img=img, adata=count_area_filtered_adata,output_name=dir_base+"F4e-3.pre.tiff")

#脉络丛上皮细胞
plot_all_area_gene_overlay(title="Complete histological image-prediction",gdf=gdf, gene='Ttr', img=img, adata=count_area_filtered_adata,output_name=dir_base+"F4e-4.pre.tiff")

#多细胞marker且重要
plot_all_area_gene_overlay(title="Complete histological image-prediction",gdf=gdf, gene='Apoe', img=img, adata=count_area_filtered_adata,output_name=dir_base+"F4e-5.pre.tiff")



