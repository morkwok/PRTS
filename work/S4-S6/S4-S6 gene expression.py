import os
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import scanpy as sc
from matplotlib.colors import ListedColormap
from shapely.geometry import Polygon
from tqdm import tqdm


# 配置参数
dir_base_img = 'E:/SCI/002/picture/S4/'  # 输出目录
gdf = gpd.read_file('E:/pythonproject/project002-2/area-Frozen.shp')  # 空间数据
count_area_filtered_adata = sc.read_h5ad('E:/pythonproject/project002-2/BrainHD_results-Frozen.h5ad')  # 原始数据
count_area_filtered_adata_pre = sc.read_h5ad('E:/SCI/002/data/prediction/frozen-prediction.h5ad')  # 预测数据



# Calculate quality control metrics for the filtered AnnData object
# 计算过滤后的AnnData对象的质量控制度量
sc.pp.calculate_qc_metrics(count_area_filtered_adata, inplace=True)

# Normalize total counts for each cell in the AnnData object
# 规范化AnnData对象中每个单元格的总数
sc.pp.normalize_total(count_area_filtered_adata, inplace=True)

# Logarithmize the values in the AnnData object after normalization
# 在归一化之后对AnnData对象中的值取对数
sc.pp.log1p(count_area_filtered_adata)



# Calculate quality control metrics for the filtered AnnData object
# 计算过滤后的AnnData对象的质量控制度量
sc.pp.calculate_qc_metrics(count_area_filtered_adata_pre, inplace=True)

# Normalize total counts for each cell in the AnnData object
# 规范化AnnData对象中每个单元格的总数
sc.pp.normalize_total(count_area_filtered_adata_pre, inplace=True)

# Logarithmize the values in the AnnData object after normalization
# 在归一化之后对AnnData对象中的值取对数
sc.pp.log1p(count_area_filtered_adata_pre)




def plot_combined_genes(gene):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # 统一色阶范围
    vmin = min(count_area_filtered_adata[:, gene].X.min(),
               count_area_filtered_adata_pre[:, gene].X.min())
    vmax = max(count_area_filtered_adata[:, gene].X.max(),
               count_area_filtered_adata_pre[:, gene].X.max())
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.PuRd  # 颜色映射

    # 绘图
    for idx, adata in enumerate([count_area_filtered_adata, count_area_filtered_adata_pre]):
        # 合并基因表达数据
        gene_df = adata[:, gene].to_df()
        gene_df['id'] = gene_df.index
        merged_gdf = gdf.merge(gene_df, left_on='id', right_on='id')

        # 生成颜色映射
        colors_main = cmap(norm(merged_gdf[gene].values))
        colors_main[:, 3] = 1  # 主体层不透明

        # 动态透明度映射函数
        def dynamic_alpha(norm_values,
                          min_alpha=0,
                          max_alpha=0.8):
            return min_alpha + (max_alpha - min_alpha) * norm_values

        # 光晕层颜色
        colors_halo = cmap(norm(merged_gdf[gene].values))
        colors_halo[:, 3] = dynamic_alpha(norm(merged_gdf[gene].values))

        # 绘制光晕层
        dilated_geoms = merged_gdf.geometry.buffer(100, resolution=16)
        gpd.GeoDataFrame(geometry=dilated_geoms).plot(
            color=colors_halo,
            edgecolor='none',
            ax=axes[idx]
        )

        # 绘制主体层
        merged_gdf.plot(
            color=colors_main,
            edgecolor='none',
            ax=axes[idx]
        )

        # 设置子图属性
        axes[idx].set_title(f"{gene} ({'Original' if idx == 0 else 'Predicted'})")
        axes[idx].axis('off')  # 隐藏坐标轴

    # 添加统一颜色条
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                 cax=cbar_ax, label='Expression Level')

    # 保存图像
    plt.savefig(f"{dir_base_img}/{gene}_combined.tiff", dpi=500, bbox_inches='tight')
    plt.close()


# 批量处理基因列表
gene_list = pd.read_csv('E:/SCI/002/picture/S4/gene_list.csv')['gene_name'].tolist()
for gene in tqdm(gene_list):
    plot_combined_genes(gene)