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

def plot_clusters_and_save_image(title, gdf, img, adata, bbox=None, color_by_obs=None, output_name=None, color_list=None):
    color_list=["#7f0000","#808000","#483d8b","#008000","#bc8f8f","#008b8b","#4682b4","#000080","#d2691e","#9acd32","#8fbc8f","#800080","#b03060","#ff4500","#ffa500","#ffff00","#00ff00","#8a2be2","#00ff7f","#dc143c","#00ffff","#0000ff","#ff00ff","#1e90ff","#f0e68c","#90ee90","#add8e6","#ff1493","#7b68ee","#ee82ee"]
    if bbox is not None:
        cropped_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    else:
        cropped_img = img

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(cropped_img, cmap='gray', origin='lower')
    axes[0].set_title(title)
    axes[0].axis('off')

    if bbox is not None:
        bbox_polygon = Polygon([(bbox[0], bbox[1]), (bbox[2], bbox[1]), (bbox[2], bbox[3]), (bbox[0], bbox[3])])

    unique_values = adata.obs[color_by_obs].astype('category').cat.categories
    num_categories = len(unique_values)

    if color_list is not None and len(color_list) >= num_categories:
        custom_cmap = ListedColormap(color_list[:num_categories], name='custom_cmap')
    else:
        # Use default tab20 colors if color_list is insufficient
        tab20_colors = plt.cm.tab20.colors[:num_categories]
        custom_cmap = ListedColormap(tab20_colors, name='custom_tab20_cmap')

    merged_gdf = gdf.merge(adata.obs[color_by_obs].astype('category'), left_on='id', right_index=True)

    if bbox is not None:
        intersects_bbox = merged_gdf['geometry'].intersects(bbox_polygon)
        filtered_gdf = merged_gdf[intersects_bbox]
    else:
        filtered_gdf = merged_gdf

    # Plot the filtered polygons on the second axis
    plot = filtered_gdf.plot(column=color_by_obs, cmap=custom_cmap, ax=axes[1], legend=True)
    axes[1].set_title(color_by_obs)
    legend = axes[1].get_legend()
    legend.set_bbox_to_anchor((1.05, 1))
    axes[1].axis('off')

    # Move legend outside the plot
    plot.get_legend().set_bbox_to_anchor((1.25, 1))

    if output_name is not None:
        plt.savefig(output_name, bbox_inches='tight')
    else:
        plt.show()

# Plotting function for nuclei area distribution 核面积分布的绘图函数
def plot_nuclei_area(gdf,area_cut_off):
    fig, axs = plt.subplots(1, 2, figsize=(15, 4))
    # Plot the histograms
    axs[0].hist(gdf['area'], bins=50, edgecolor='black')
    axs[0].set_title('Nuclei Area')

    axs[1].hist(gdf[gdf['area'] < area_cut_off]['area'], bins=50, edgecolor='black')
    axs[1].set_title('Nuclei Area Filtered:'+str(area_cut_off))

    plt.tight_layout()
    plt.show()

# Total UMI distribution plotting function 总UMI分布绘图函数
def total_umi(adata_, cut_off):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # Box plot
    axs[0].boxplot(adata_.obs["total_counts"], vert=False, widths=0.7, patch_artist=True, boxprops=dict(facecolor='skyblue'))
    axs[0].set_title('Total Counts')

    # Box plot after filtering
    axs[1].boxplot(adata_.obs["total_counts"][adata_.obs["total_counts"] > cut_off], vert=False, widths=0.7, patch_artist=True, boxprops=dict(facecolor='skyblue'))
    axs[1].set_title('Total Counts > ' + str(cut_off))

    # Remove y-axis ticks and labels
    for ax in axs:
        ax.get_yaxis().set_visible(False)

    plt.tight_layout()
    plt.show()


sc.settings.set_figure_params(dpi=50, dpi_save=300, figsize=(7, 7))

import anndata

dir_base = 'E:/SCI/002/data/braintrain/BrainHD/Visium_HD_Mouse_Brain_binned_outputs/binned_outputs/square_002um/'
#图像的文件名
filename = 'Visium_HD_Mouse_Brain_tissue_image.tif'
img = imread(dir_base + filename)

#细胞坐标信息
import geopandas as gpd
gdf = gpd.read_file("E:/pythonproject/project002-2/area-input2.shp")


gdf.head()


# Define a single color cmap 定义一个单色cmap
cmap=ListedColormap(['grey'])

# Create Plot 创建图
#plot_mask_and_save_image(title="TRY-Region of Interest 1",gdf=gdf,bbox=(12844,7700,13760,8664),cmap=cmap,img=img,output_name=dir_base+"002image_1-mask.ROI1.tif")


#plot_mask_and_save_image(title="TRY-Region of Interest 2",gdf=gdf,cmap=cmap,img=img,output_name=dir_base+"002image_ALL-mask.ROI2.tif")




###汇总Visium HD基因表达数据
# Load Visium HD data 加载Visium HD数据
raw_h5_file = dir_base+'filtered_feature_bc_matrix.h5'
adata = sc.read_10x_h5(raw_h5_file)

# Load the Spatial Coordinates 加载空间坐标
tissue_position_file = dir_base+'tissue_positions.parquet'
df_tissue_positions=pd.read_parquet(tissue_position_file)

#Set the index of the dataframe to the barcodes 将数据帧的索引设置为条形码
df_tissue_positions = df_tissue_positions.set_index('barcode')

# Create an index in the dataframe to check joins 在数据框架中创建索引以检查连接
df_tissue_positions['index']=df_tissue_positions.index

# Adding the tissue positions to the meta data 将组织位置添加到元数据
adata.obs =  pd.merge(adata.obs, df_tissue_positions, left_index=True, right_index=True)

# Create a GeoDataFrame from the DataFrame of coordinates 从坐标的DataFrame创建一个GeoDataFrame
geometry = [Point(xy) for xy in zip(df_tissue_positions['pxl_col_in_fullres'], df_tissue_positions['pxl_row_in_fullres'])]
gdf_coordinates = gpd.GeoDataFrame(df_tissue_positions, geometry=geometry)

#检查每个条形码，以确定它们是否在细胞核中.确保每个条形码唯一分配
# Perform a spatial join to check which coordinates are in a cell nucleus 执行空间连接以检查哪些坐标位于细胞核中
result_spatial_join = gpd.sjoin(gdf_coordinates, gdf, how='left', predicate='within')

# Identify nuclei associated barcodes and find barcodes that are in more than one nucleus 识别核相关的条形码和发现条形码在多个核的情况
result_spatial_join['is_within_polygon'] = ~result_spatial_join['index_right'].isna()
barcodes_in_overlaping_polygons = pd.unique(result_spatial_join[result_spatial_join.duplicated(subset=['index'])]['index'])
result_spatial_join['is_not_in_an_polygon_overlap'] = ~result_spatial_join['index'].isin(barcodes_in_overlaping_polygons)


Result = result_spatial_join.loc[result_spatial_join['is_within_polygon']]
Result[['index', 'id', 'is_within_polygon', 'is_not_in_an_polygon_overlap']].to_csv("Nuclei_Barcode_Map.csv")



# Remove barcodes in overlapping nuclei 去除重叠核中的条形码
barcodes_in_one_polygon = result_spatial_join[result_spatial_join['is_within_polygon'] & result_spatial_join['is_not_in_an_polygon_overlap']]

# The AnnData object is filtered to only contain the barcodes that are in non-overlapping polygon regions AnnData对象被过滤为只包含非重叠多边形区域中的条形码
filtered_obs_mask = adata.obs_names.isin(barcodes_in_one_polygon['index'])
filtered_adata = adata[filtered_obs_mask,:]

# Add the results of the point spatial join to the Anndata object 将点空间连接的结果添加到Anndata对象
filtered_adata.obs =  pd.merge(filtered_adata.obs, barcodes_in_one_polygon[['index','geometry','id','is_within_polygon','is_not_in_an_polygon_overlap']], left_index=True, right_index=True)

#对自定义装箱数据执行一个基因计数求和
# Group the data by unique nucleous IDs 按唯一的核id分组数据
groupby_object = filtered_adata.obs.groupby(['id'], observed=True)

# Extract the gene expression counts from the AnnData object 从AnnData对象中提取基因表达计数
counts = filtered_adata.X

# Obtain the number of unique nuclei and the number of genes in the expression data 获得唯一细胞核数和表达数据中基因数
N_groups = groupby_object.ngroups
N_genes = counts.shape[1]

# Initialize a sparse matrix to store the summed gene counts for each nucleus 初始化一个稀疏矩阵来存储每个细胞核的总和基因计数
summed_counts = sparse.lil_matrix((N_groups, N_genes))

# Lists to store the IDs of polygons and the current row index 用于存储多边形id和当前行索引的列表
polygon_id = []
row = 0

# Iterate over each unique polygon to calculate the sum of gene counts. 遍历每个唯一的多边形以计算基因计数的总和。
for polygons, idx_ in groupby_object.indices.items():
    summed_counts[row] = counts[idx_].sum(0)
    row += 1
    polygon_id.append(polygons)

# Create and AnnData object from the summed count matrix 从求和计数矩阵创建和AnnData对象
summed_counts = summed_counts.tocsr()
grouped_filtered_adata = anndata.AnnData(X=summed_counts,obs=pd.DataFrame(polygon_id,columns=['id'],index=polygon_id),var=filtered_adata.var)

#%store grouped_filtered_adata

#质控（去除非常大的核和UMI计数过少的核）
# Store the area of each nucleus in the GeoDataframe 将每个核的区域存储在GeoDataframe中
gdf['area'] = gdf['geometry'].area

# Calculate quality control metrics for the original AnnData object 计算原始AnnData对象的质量控制度量
sc.pp.calculate_qc_metrics(grouped_filtered_adata, inplace=True)

# Plot the nuclei area distribution before and after filtering 绘制过滤前后的核面积分布
#根据核分布，我们选择了一个值为2000来过滤数据。
plot_nuclei_area(gdf=gdf,area_cut_off=2000)

# Plot total UMI distribution 小区总UMI分布 总计20个UMI的阈值
total_umi(grouped_filtered_adata, 20)


#
# Create a mask based on the 'id' column for values present in 'gdf' with 'area' less than 2000
#为“gdf”中“area”小于2000的值创建一个基于“id”列的蒙版
mask_area = grouped_filtered_adata.obs['id'].isin(gdf[gdf['area'] < 2000].id)

# Create a mask based on the 'total_counts' column for values greater than 20
# 为大于20的值创建基于‘total_counts’列的掩码
mask_count = grouped_filtered_adata.obs['total_counts'] > 20

# Apply both masks to the original AnnData to create a new filtered AnnData object
# 将两个掩码应用到原始的AnnData上，以创建一个新的过滤后的AnnData对象
count_area_filtered_adata = grouped_filtered_adata[mask_area & mask_count, :]


count_area_filtered_adata.var_names_make_unique()

count_area_filtered_adata.var['mt'] = count_area_filtered_adata.var_names.str.startswith('mt-')  # annotate the group of mitochondrial genes as 'mt'
sc.pp.calculate_qc_metrics(count_area_filtered_adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
sc.pl.violin(count_area_filtered_adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
             jitter=0.4, multi_panel=True)

# 提取线粒体dna在15%以下
count_area_filtered_adata = count_area_filtered_adata[count_area_filtered_adata.obs.pct_counts_mt < 15, :]




count_area_filtered_adata.write('BrainHD_results-input1.h5ad', compression="gzip")
