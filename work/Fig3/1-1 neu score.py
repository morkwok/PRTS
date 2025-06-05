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


def plot_score_and_save_image(title, gdf, img, adata, bbox=None, output_name=None):

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

    # 提取 obs 中的 score 信息并合并到地理数据框架
    score_info = adata.obs[['score']].copy()  # 提取 score 列（保留样本索引）
    score_info['id'] = score_info.index  # 将索引转换为 id 列
    merged_gdf = gdf.merge(score_info, left_on='id', right_on='id')  # 按 id 合并


    if bbox is not None:
        # Filter for polygons in the box
        intersects_bbox = merged_gdf['geometry'].intersects(bbox_polygon)
        filtered_gdf = merged_gdf[intersects_bbox]
    else:
        filtered_gdf = merged_gdf

    # 设置归一化参数（使用score列）[5](@ref)
    vmin = filtered_gdf['score'].quantile(0.01)
    vmax = filtered_gdf['score'].quantile(0.99)
    norm = plt.Normalize(vmin=vmin, vmax=vmax, clip=True)
    # 颜色映射设置（可替换为其他cmap）[8](@ref)
    cmap = plt.cm.coolwarm  # 保持原始配色
    # 动态透明度参数[7](@ref)
    alpha_halo = (0.2, 0.6)  # 缓冲层透明度
    alpha_main = (0.7, 1.0)  # 主体层透明度

    # 生成动态RGBA颜色[3](@ref)
    def generate_alpha_colors(values, cmap, alpha_range):
        """生成带动态透明度的颜色矩阵"""
        rgba = cmap(values)
        alpha = alpha_range[0] + (alpha_range[1] - alpha_range[0]) * values
        rgba[:, 3] = alpha
        return rgba

    # 创建缓冲几何体（调整buffer参数）[5](@ref)
    buffered_geoms = filtered_gdf.geometry.buffer(
        distance=80,  # 膨胀距离（单位与坐标系相关）
        resolution=16  # 圆弧分段数
    )
    buffered_gdf = gpd.GeoDataFrame(filtered_gdf, geometry=buffered_geoms)

    # 子图2：分数可视化
    ax = axes[1]
    norm_values = norm(filtered_gdf['score'].values)

    # 先绘制缓冲层[5](@ref)
    buffered_gdf.plot(
        color=generate_alpha_colors(norm_values, cmap, alpha_halo),
        edgecolor='none',
        ax=ax
    )

    # 再绘制主体层[7](@ref)
    filtered_gdf.plot(
        color=generate_alpha_colors(norm_values, cmap, alpha_main),
        edgecolor='none',  # 添加白色边界增强对比[8](@ref)
        linewidth=0.8,
        ax=ax
    )

    # 添加颜色条[1](@ref)
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        shrink=0.8,
        label='Marker Score'
    )
    ax.set_title("Spatial Score Distribution")
    ax.axis('off')

    # 保存或显示结果[4](@ref)
    if output_name:
        plt.savefig(output_name, bbox_inches='tight', dpi=300, transparent=True)  # 保存透明背景[4](@ref)
    else:
        plt.show()

    plt.close()



raw_h5_file = 'E:/SCI/002/data/prediction/frozen-prediction.h5ad'
count_area_filtered_adata = sc.read_h5ad(raw_h5_file)



dir_base = 'E:/SCI/002/data/brainfixfrozen/Visium_HD_Mouse_Brain_Fixed_Frozen_binned_outputs/binned_outputs/square_002um/'
#图像的文件名
filename = 'Visium_HD_Mouse_Brain_Fixed_Frozen_tissue_image.btf'
img = imread(dir_base + filename)

#细胞坐标信息

import geopandas as gpd
gdf = gpd.read_file("E:/pythonproject/project002-2/area-Frozen.shp")

gdf.head()

dir_base = 'E:/SCI/002/picture/S7/'

# Calculate quality control metrics for the filtered AnnData object
# 计算过滤后的AnnData对象的质量控制度量
sc.pp.calculate_qc_metrics(count_area_filtered_adata, inplace=True)

# Normalize total counts for each cell in the AnnData object
# 规范化AnnData对象中每个单元格的总数
sc.pp.normalize_total(count_area_filtered_adata, inplace=True)

# Logarithmize the values in the AnnData object after normalization
# 在归一化之后对AnnData对象中的值取对数
sc.pp.log1p(count_area_filtered_adata)

# '0': 'Neu_01_Lrp1b',
gene_set=["Lrp1b","Ntrk3","Trp53bp1","Pitpnm2",'Slf2']
sc.tl.score_genes(count_area_filtered_adata,gene_set)
plot_score_and_save_image(title="Neu_01 score", gdf=gdf, img=img, adata=count_area_filtered_adata,output_name=dir_base+"Neu_01-pre.tiff")

#'2': 'Neu_02_Stip1',
gene_set=["Stip1","Pitpnm1","Agbl4","Dpysl2",'Snhg11']
sc.tl.score_genes(count_area_filtered_adata,gene_set)
plot_score_and_save_image(title="Neu_02 score", gdf=gdf, img=img, adata=count_area_filtered_adata,output_name=dir_base+"Neu_02-pre.tiff")


#'4': 'Neu_03_Cacnb3',
gene_set=["Cacnb3","Col25a1","Tspan13","Tesc",'Syt13']
sc.tl.score_genes(count_area_filtered_adata,gene_set)
plot_score_and_save_image(title="Neu_03 score", gdf=gdf, img=img, adata=count_area_filtered_adata,output_name=dir_base+"Neu_03-pre.tiff")

#'6': 'Neu_04_Lrrtm4',
gene_set=["Ncdn","Ccdc85a","Lrrtm4","Plk5",'Epha7']
sc.tl.score_genes(count_area_filtered_adata,gene_set)
plot_score_and_save_image(title="Neu_04 score", gdf=gdf, img=img, adata=count_area_filtered_adata,output_name=dir_base+"Neu_04-pre.tiff")

#'7': 'Neu_05_Ncdn',
gene_set=["Ncdn","Bhlhe22","Fam163b","Pcdh20",'Epha7']
sc.tl.score_genes(count_area_filtered_adata,gene_set)
plot_score_and_save_image(title="Neu_05 score", gdf=gdf, img=img, adata=count_area_filtered_adata,output_name=dir_base+"Neu_05-pre.tiff")

# '12': 'Neu_06_Cbln1',,
gene_set=["Cbln1","Cbln4","Cbln2","Slc6a11",'Calb2']
sc.tl.score_genes(count_area_filtered_adata,gene_set)
plot_score_and_save_image(title="Neu_06 score", gdf=gdf, img=img, adata=count_area_filtered_adata,output_name=dir_base+"Neu_06-pre.tiff")

# '13': 'Neu_07_Slc7a14',
gene_set=["Ppp3ca","Slc7a14","Tafa2","Arhgef25",'Plxna4']
sc.tl.score_genes(count_area_filtered_adata,gene_set)
plot_score_and_save_image(title="Neu_07 score", gdf=gdf, img=img, adata=count_area_filtered_adata,output_name=dir_base+"Neu_07-pre.tiff")

#  '14': 'Neu_08_Slc6a11',
gene_set=["C1ql2","Shisal1","Cygb","Slc6a11",'Nrp2']
sc.tl.score_genes(count_area_filtered_adata,gene_set)
plot_score_and_save_image(title="Neu_08 score", gdf=gdf, img=img, adata=count_area_filtered_adata,output_name=dir_base+"Neu_08-pre.tiff")

#  '15': 'Neu_09',
gene_set=["Neurod6","Satb2","Vxn","Dkkl1",'Dusp6']
sc.tl.score_genes(count_area_filtered_adata,gene_set)
plot_score_and_save_image(title="Neu_09 score", gdf=gdf, img=img, adata=count_area_filtered_adata,output_name=dir_base+"Neu_09-pre.tiff")


# '17': 'Neu_10_Ppp1r9a',
gene_set=["Ppp1r9a","Ndrg4","Jdp2","Gprin1",'Lzts3']
sc.tl.score_genes(count_area_filtered_adata,gene_set)
plot_score_and_save_image(title="Neu_10 score", gdf=gdf, img=img, adata=count_area_filtered_adata,output_name=dir_base+"Neu_10-pre.tiff")


#  '18': 'Neu_11_Snrpn',
gene_set=["Snrpn","Tspyl4","Ndfip1","Atp6v0a1",'Evl']
sc.tl.score_genes(count_area_filtered_adata,gene_set)
plot_score_and_save_image(title="Neu_11 score", gdf=gdf, img=img, adata=count_area_filtered_adata,output_name=dir_base+"Neu_11-pre.tiff")


#  '19': 'Neu_12_Sema5a',
gene_set=["Orai2","Sema5a","Synpr","Neurod1",'C1ql2']
sc.tl.score_genes(count_area_filtered_adata,gene_set)
plot_score_and_save_image(title="Neu_12 score", gdf=gdf, img=img, adata=count_area_filtered_adata,output_name=dir_base+"Neu_12-pre.tiff")

# '20': 'Neu_13_Cbln4',
gene_set=["C1ql2","Prox1","Cbln4","Grm2",'Calb2']
sc.tl.score_genes(count_area_filtered_adata,gene_set)
plot_score_and_save_image(title="Neu_13 score", gdf=gdf, img=img, adata=count_area_filtered_adata,output_name=dir_base+"Neu_13-pre.tiff")


#  '21': 'Neu_14_Prox1'
gene_set=["C1ql2","Orai2","Prox1","Cygb",'Zic1']
sc.tl.score_genes(count_area_filtered_adata,gene_set)
plot_score_and_save_image(title="Neu_14 score", gdf=gdf, img=img, adata=count_area_filtered_adata,output_name=dir_base+"Neu_14-pre.tiff")