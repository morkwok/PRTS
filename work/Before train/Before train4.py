import anndata
import pandas as pd
import scanpy as sc
#import omicverse as ov

sc.settings.set_figure_params(dpi=50, dpi_save=300, figsize=(7, 7))

dir_base = 'E:/pythonproject/project002-2/'

###汇总Visium HD基因表达数据
# Load Visium HD data 加载Visium HD数据
raw_h5_file = dir_base+'Brain-HD_results-cluster.h5ad'
mydata = sc.read_h5ad(raw_h5_file)


mydata.var_names_make_unique()

#要保留
sc.tl.rank_genes_groups(mydata, groupby='clusters1', method='wilcoxon')
sc.pl.rank_genes_groups(mydata, n_genes=25, sharey=False)


neuro_marker_genes_dict = {
    'neuron': ['Snap25', 'Camk2n1','Ncdn','Atp1b1'],#神经细胞 #cluster 0、1、6、12、15
}

sc.pl.dotplot(mydata, neuro_marker_genes_dict, 'clusters1', standard_scale="var",dendrogram=True)

sc.pl.dotplot(mydata, neuro_marker_genes_dict, 'clusters1',dendrogram=True)




neuroglial_marker_genes_dict = {
    'oligodendrocyte': ['Plp1', 'Mbp','Mobp'],#少突胶质细胞 #cluster 3
    'Astrocyte': ['Apoe','Aldoc','Clu'],#星形胶质细胞 #cluster 2、4、5、7、8、9、11、13、17
}

sc.pl.dotplot(mydata, neuroglial_marker_genes_dict, 'clusters1', standard_scale="var",dendrogram=True)

sc.pl.dotplot(mydata, neuroglial_marker_genes_dict, 'clusters1',dendrogram=True)


cpe_marker_genes_dict = {
    'Choroid plexis epithelial': ['Ttr', 'Enpp2'],  # 脉络丛上皮细胞 #cluster 16
}

sc.pl.dotplot(mydata, cpe_marker_genes_dict, 'clusters1', standard_scale="var",dendrogram=True)

sc.pl.dotplot(mydata, cpe_marker_genes_dict, 'clusters1',dendrogram=True)

#cluster 10、14、18、19 low expression。

rank_results = mydata.uns['rank_genes_groups']
print(rank_results)

all_groups_results = pd.DataFrame()

group_labels = mydata.obs[rank_results['params']['groupby']].unique()
for group_label in group_labels:
    group_df = sc.get.rank_genes_groups_df(mydata, group=group_label)
    group_df = group_df.sort_values(by="scores", ascending=False)
    group_df = group_df[
        (abs(group_df['logfoldchanges']) > 1) &
        (group_df['pvals_adj'] < 0.01)  # 假设列名为pvals_adj
        ]
    group_df['group'] = group_label
    all_groups_results = pd.concat([all_groups_results, group_df], ignore_index=True)

all_groups_results.to_csv('differential_expression_all_groups-train.csv', index=False)


# 提取唯一基因名（去重）
unique_genes = all_groups_results['names'].drop_duplicates().reset_index(drop=True)  # :ml-citation{ref="4,6" data="citationList"}
# 导出为CSV文件
unique_genes.to_csv('train_genes.csv', index=False, header=['gene'])




cluster2annotation = {
    '0': 'neuron',
    '1': 'neuron',
    '2': 'Astrocyte',
    '3': 'oligodendrocyte',
    '4': 'Astrocyte',
    '5': 'Astrocyte',
    '6': 'neuron',
    '7': 'Astrocyte',
    '8': 'Astrocyte',
    '9': 'Astrocyte',
    '10': 'low expression',
    '11': 'Astrocyte',
    '12': 'neuron',
    '13': 'Astrocyte',
    '14': 'low expression',
    '15': 'neuron',
    '16': 'Choroid plexis epithelial',
    '17': 'Astrocyte',
    '18': 'low expression',
    '19': 'low expression',
}
mydata.obs['major_celltype'] = mydata.obs['clusters1'].map(cluster2annotation).astype('category')

#print(mydata.obs)


marker_genes_dict = {
    'neuron': ['Snap25', 'Camk2n1','Ncdn','Atp1b1'],#神经细胞 #聚类0、1、6、12、15
    'oligodendrocyte': ['Plp1', 'Mbp','Mobp'],#少突胶质细胞 #聚类3
    'Astrocyte': ['Apoe','Aldoc','Clu'],#星形胶质细胞 #聚类2、4、5、7、8、9、11、13、17
    'Choroid plexis epithelial': ['Ttr', 'Enpp2'] # 脉络丛上皮细胞 #聚类16
}
sc.pl.dotplot(mydata, marker_genes_dict, 'major_celltype', standard_scale="var",dendrogram=True)

sc.pl.umap(mydata, color='major_celltype',save = "S1h.png")






cluster3annotation = {
    '0': 'neuron1',
    '1': 'neuron2',
    '2': 'Astrocyte1',
    '3': 'oligodendrocyte',
    '4': 'Astrocyte3',
    '5': 'Astrocyte4',
    '6': 'neuron3',
    '7': 'Astrocyte5',
    '8': 'Astrocyte6',
    '9': 'Astrocyte7',
    '10': 'low expression',
    '11': 'Astrocyte8',
    '12': 'neuron4',
    '13': 'Astrocyte9',
    '14': 'low expression',
    '15': 'neuron5',
    '16': 'Choroid plexis epithelial',
    '17': 'Astrocyte10',
    '18': 'low expression',
    '19': 'low expression',
}
mydata.obs['second_celltype'] = mydata.obs['clusters1'].map(cluster3annotation).astype('category')


sc.pl.umap(mydata, color='second_celltype',save = "test2.heatmap.png")






#导入模块

import matplotlib.pyplot as plt
import geopandas as gpd

from tifffile import imread, imwrite
from shapely.geometry import Polygon, Point
from matplotlib.colors import ListedColormap


sc.settings.set_figure_params(dpi=50, dpi_save=300, figsize=(7, 7))

dir_base = 'E:/SCI/002/data/braintrain/BrainHD/Visium_HD_Mouse_Brain_binned_outputs/binned_outputs/square_002um/'
#图像的文件名
filename = 'Visium_HD_Mouse_Brain_tissue_image.tif'
img = imread(dir_base + filename)


#细胞坐标信息
gdf = gpd.read_file("E:/pythonproject/project002-2/area-input2.shp")

from matplotlib.patches import Patch
from matplotlib.lines import Line2D

def plot_clusters_and_save_image(title, gdf, img, adata, bbox=None, color_by_obs=None, output_name=None):
    # 自定义颜色列表 (30种可区分颜色)
    color_list = [
        "#7f0000", "#808000", "#483d8b", "#008000", "#bc8f8f",
        "#008b8b", "#4682b4", "#000080", "#d2691e", "#9acd32",
        "#8fbc8f", "#800080", "#b03060", "#ff4500", "#ffa500",
        "#ffff00", "#00ff00", "#8a2be2", "#00ff7f", "#dc143c",
        "#00ffff", "#0000ff", "#ff00ff", "#1e90ff", "#f0e68c",
        "#90ee90", "#add8e6", "#ff1493", "#7b68ee", "#ee82ee"
    ]

    # 创建画布和子图
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    plt.subplots_adjust(wspace=0.3, right=0.72)

    # ================= 处理原始图像 =================
    if bbox is not None:
        xmin, ymin, xmax, ymax = bbox
        cropped_img = img[ymin:ymax, xmin:xmax]
        bbox_poly = Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])
    else:
        cropped_img = img
        bbox_poly = None

    # 绘制原始图像
    axes[0].imshow(cropped_img, cmap='gray', origin='lower')
    axes[0].set_title(f"{title}\nOriginal Image", fontsize=12)
    axes[0].axis('off')

    # ================= 处理聚类数据 =================
    # 合并空间数据与观测数据
    merged_gdf = gdf.merge(
        adata.obs[[color_by_obs]],
        left_on='id',
        right_index=True
    )

    # 筛选感兴趣区域
    if bbox_poly is not None:
        intersects = merged_gdf.geometry.intersects(bbox_poly)
        filtered_gdf = merged_gdf[intersects].copy()
    else:
        filtered_gdf = merged_gdf.copy()

    # ================= 颜色映射处理 =================
    # 获取全局类别信息
    all_categories = sorted(adata.obs[color_by_obs].astype(str).unique())
    n_cats = len(all_categories)

    # 生成扩展颜色列表
    extended_colors = (color_list * (n_cats // len(color_list) + 1))[:n_cats]
    color_dict = {cat: extended_colors[i] for i, cat in enumerate(all_categories)}

    # 转换为分类变量以保持顺序
    filtered_gdf[color_by_obs] = pd.Categorical(
        filtered_gdf[color_by_obs].astype(str),
        categories=all_categories
    )

    # ================= 绘制聚类结果 =================
    # 创建自定义颜色映射
    cmap = ListedColormap(extended_colors)

    # 绘制地理空间数据
    plot = filtered_gdf.plot(
        column=color_by_obs,
        cmap=cmap,
        ax=axes[1],
        legend=False
    )

    # ================= 创建自定义图例 =================
    # 生成圆形图例句柄
    legend_elements = [
        Line2D(
            [0], [0],
            marker='o',  # 圆形标记
            color='w',  # 边框颜色（白色不可见）
            markerfacecolor=color_dict[cat],  # 填充颜色
            markersize=12,  # 标记大小
            linestyle='none',  # 不显示连接线
            label=cat  # 类别标签
        )
        for cat in all_categories
        if cat in filtered_gdf[color_by_obs].values
    ]

    # 添加图例
    axes[1].legend(
        handles=legend_elements,
        bbox_to_anchor=(1.28, 0.5),  # 图例位置
        loc='center left',
        frameon=False,  # 无边框
        title=color_by_obs,  # 图例标题
        title_fontsize=12,  # 标题字号
        fontsize=10,  # 标签字号
        labelspacing=1.2  # 标签间距
    )

    # 设置子图属性
    axes[1].set_title(f"{title}\nClustering Result", fontsize=12)
    axes[1].axis('off')

    # ================= 保存结果 =================
    if output_name:
        plt.savefig(
            output_name,
            bbox_inches='tight',  # 紧凑布局
            dpi=300,  # 高分辨率
            pad_inches=0.8,  # 边缘留白
            facecolor='white'  # 背景颜色
        )
    plt.close()

# 绘制并保存聚类结果
plot_clusters_and_save_image(title="Region of interest 1", gdf=gdf, img=img, adata=mydata, bbox=(12844,7700,13760,8664), color_by_obs='major_celltype', output_name=dir_base+"S1J-1.tiff")
plot_clusters_and_save_image(title="Region of interest 2", gdf=gdf, img=img, adata=mydata, bbox=(16844,7700,17760,8664), color_by_obs='major_celltype', output_name=dir_base+"s1J-2.tiff")

plot_clusters_and_save_image(title="Hippocampus", gdf=gdf, img=img, adata=mydata, bbox=(9844,4700,19760,9664), color_by_obs='major_celltype', output_name=dir_base+"S1k.tiff")


plot_clusters_and_save_image(title="Complete histological image", gdf=gdf, img=img, adata=mydata, color_by_obs='major_celltype', output_name=dir_base+"S1I.tiff")
plot_clusters_and_save_image(title="Complete histological image", gdf=gdf, img=img, adata=mydata, color_by_obs='second_celltype', output_name=dir_base+"image9_clustering.tiff")





mydata.write('Brain-HD_results-annotation.h5ad', compression="gzip")

print(mydata)
print("注释结果:",mydata.obs)

