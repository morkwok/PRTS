#导入模块
import matplotlib.pyplot as plt
import scanpy as sc
from tifffile import imread
from shapely.geometry import Polygon
from matplotlib.colors import ListedColormap





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
    # 计算features
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
    vmin = filtered_gdf['total_counts'].quantile(0.01)
    vmax = filtered_gdf['total_counts'].quantile(0.99)
    norm = plt.Normalize(vmin=vmin, vmax=vmax, clip=True)  # 启用裁剪
    cmap = plt.cm.PuBu

    # 生成强制裁剪的归一化值
    raw_values = filtered_gdf['total_counts'].values
    norm_values = norm(raw_values)

    # 验证归一化范围（可选）
    print(f"归一化值范围: {norm_values.min():.2f}-{norm_values.max():.2f}")  # 应输出0.00-1.00
    # 设置透明度参数
    alpha_halo = (0.2, 0.8)  # 缓冲层透明度范围（低值透明，高值不透明）
    alpha_main = (0.8, 1.0)  # 主体层透明度范围

    # 动态生成RGBA颜色数组
    def generate_alpha_colors(values, cmap, alpha_range):
        """生成带动态透明度的颜色矩阵"""
        rgba = cmap(values)  # 输入应为0-1的数值数组
        alpha = alpha_range[0] + (alpha_range[1] - alpha_range[0]) * values
        rgba[:, 3] = alpha
        return rgba

    # 创建缓冲几何体（调整buffer参数控制膨胀程度）
    buffered_geoms = filtered_gdf.geometry.buffer(
        distance=100,  # 膨胀距离（单位与CRS相关，经纬度需投影转换）
        resolution=16  # 圆弧分段数（越高越平滑）
    )
    buffered_gdf = gpd.GeoDataFrame(filtered_gdf, geometry=buffered_geoms)

    # 绘制到第二个子图
    ax = axes[1]

    # 先绘制缓冲层（半透明外圈）
    buffered_gdf.plot(
        color=generate_alpha_colors(norm_values, cmap, alpha_halo),
        edgecolor='none',  # 禁用边界线
        ax=ax
    )

    # 再绘制主体层（较高不透明度）
    filtered_gdf.plot(
        color=generate_alpha_colors(norm_values, cmap, alpha_main),
        edgecolor='none',  # 添加白色边界增强对比
        linewidth=0.5,
        ax=ax
    )

    # 添加颜色条（反映颜色映射）
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        shrink=0.8,
        label=f'Total UMI Counts'
    )
    ax.set_title("Total UMI Counts")
    ax.axis('off')

    # ---- 绘制总Features ----
    vmin = filtered_gdf['n_features'].quantile(0.01)
    vmax = filtered_gdf['n_features'].quantile(0.99)
    norm = plt.Normalize(vmin=vmin, vmax=vmax, clip=True)  # 启用裁剪
    cmap = plt.cm.PuRd

    # 动态生成RGBA颜色数组
    def generate_alpha_colors(values, cmap, alpha_range):
        """生成带动态透明度的颜色矩阵"""
        rgba = cmap(values)  # 获取基础颜色
        alpha = alpha_range[0] + (alpha_range[1] - alpha_range[0]) * values
        rgba[:, 3] = alpha  # 替换Alpha通道
        return rgba

    # 创建缓冲几何体（调整buffer参数控制膨胀程度）
    buffered_geoms = filtered_gdf.geometry.buffer(
        distance=100,  # 膨胀距离（单位与CRS相关，经纬度需投影转换）
        resolution=16  # 圆弧分段数（越高越平滑）
    )
    buffered_gdf = gpd.GeoDataFrame(filtered_gdf, geometry=buffered_geoms)

    # 绘制到第二个子图
    ax = axes[2]

    # 先绘制缓冲层（半透明外圈）
    buffered_gdf.plot(
        color=generate_alpha_colors(norm_values, cmap, alpha_halo),
        edgecolor='none',  # 禁用边界线
        ax=ax
    )

    # 再绘制主体层（较高不透明度）
    filtered_gdf.plot(
        color=generate_alpha_colors(norm_values, cmap, alpha_main),
        edgecolor='none',  # 添加白色边界增强对比
        linewidth=0.5,
        ax=ax
    )

    # 添加颜色条（反映颜色映射）
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        shrink=0.8,
        label=f'Number of Features'
    )
    ax.set_title("Detected Features")
    ax.axis('off')

    # 调整布局并保存/显示
    plt.tight_layout()
    if output_name:
        plt.savefig(output_name, bbox_inches='tight', dpi=300)
    else:
        plt.show()

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
raw_h5_file = 'E:/SCI/002/data/prediction/frozen-prediction.h5ad'
count_area_filtered_adata = sc.read_h5ad(raw_h5_file)

dir_base = 'E:/SCI/002/picture/Fig2/'



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
    output_name=dir_base+"F2ab-prediction.png"
)


