import anndata
import pandas as pd
import scanpy as sc
#import omicverse as ov

#sc.settings.set_figure_params(dpi=50,frameon=False,figsize=(3,3),facecolor='white')

sc.settings.set_figure_params(dpi=50, dpi_save=300, figsize=(7, 7))


#sc.settings.set_figure_params(dpi=300,frameon=False,figsize=(3,3),facecolor='white')

dir_base = 'E:/pythonproject/project002-2/S9(BA2)/'

###汇总Visium HD基因表达数据
# Load Visium HD data 加载Visium HD数据
raw_h5_file = dir_base+'S9跑图BA2.h5ad'
mydata = sc.read_h5ad(raw_h5_file)



cluster2annotation = {
    '0': 'Neu_01',
    '1': 'OLG_01',
    '2': 'AC_01',
    '3': 'Neu_02',
    '4': 'Neu_03',
    '5': 'Neu_04',
    '6': 'Neu_05',
    '7': 'AC_02',
    '8': 'Neu_06',
    '9': 'OLG_02',
    '10': 'Neu_07',
    '11': 'Neu_08',
    '12': 'CPEC',
    '13': 'VLM',
    '14': 'Neu_09',
    '15': 'AC_03',
    '16': 'OLG_03'
}
mydata.obs['major_celltype'] = mydata.obs['clusters1'].map(cluster2annotation).astype('category')

#print(mydata.obs)

marker_genes_dict = {
    'Neu': ['Stau2', 'Stx16','Ahi1','Rab36','Tcf7l2','Zcchc12','Gap43','Camk2b','Nrgn','Hap1','Cbln1','Igf2','Ahnak'],
    'OLG': ['Plp1','Cnp','Mag','Opalin'],
    'AC': ['Apoe','Vim','Gja1','Slc6a11'],
    'CPEC': ['Ttr','Ecrg4'],
    'VLM': ['Ptgds','Mgp']
}
sc.pl.dotplot(mydata, marker_genes_dict, 'major_celltype', standard_scale="var",dendrogram=True)

#sc.pl.umap(mydata, color='major_celltype',save = "分类调试-BA1.png")


custom_order = ['Neu_01','Neu_02','Neu_03','Neu_04','Neu_05','Neu_06','Neu_07','Neu_08','Neu_09','AC_01','AC_02','AC_03',
                 'OLG_01', 'OLG_02', 'OLG_03', 'VLM', 'CPEC']  # 按需替换为实际亚群名称
mydata.obs['groupdotplot'] = pd.Categorical(
    mydata.obs['major_celltype'],
    categories=custom_order,
    ordered=True
)
sc.pl.dotplot(mydata, marker_genes_dict, 'groupdotplot',
             standard_scale="var", dendrogram=False)



# 确保导入以下模块
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


from matplotlib.lines import Line2D
import scanpy as sc
import warnings


from matplotlib.lines import Line2D

import warnings

def plot_annotated_umap(
        adata,
        cluster_key='cluster',
        style_config=None,  # 修改为 None 并合并默认配置
        save_path=None
):
    # 合并默认配置和用户传入的配置
    default_style = {
        'dot_size': 3,
        'on_data_fontsize': 6,
        'side_legend_fontsize': 9,
        'tick_fontsize': 6,
        'legend_position': (1.05, 0.5),
        'label_offset': (0.5, -0.2),  # 确保默认包含此参数
        'fig_width': 10
    }
    if style_config is None:
        style_config = {}
    style_config = {**default_style, **style_config}  # 合并字典

    # 创建画布
    fig, ax = plt.subplots(figsize=(style_config['fig_width'], 6))

    # ========== 绘制基础UMAP ==========
    sc.pl.umap(
        adata,
        color=cluster_key,
        size=style_config['dot_size'],
        alpha=0.8,
        legend_loc='on data',
        legend_fontsize=style_config['on_data_fontsize'],
        legend_fontoutline=1,
        frameon=False,
        ax=ax,
        show=False
    )

    # ========== 手动调整标签位置 ==========
    texts = [child for child in ax.get_children() if isinstance(child, plt.Text)]
    for text in texts:
        x, y = text.get_position()
        text.set_position((
            x + style_config['label_offset'][0],
            y + style_config['label_offset'][1]
        ))

    # ========== 调整坐标轴字体 ==========
    ax.set_xlabel("UMAP1", fontsize=style_config['tick_fontsize'] + 1)
    ax.set_ylabel("UMAP2", fontsize=style_config['tick_fontsize'] + 1)
    ax.tick_params(axis='both', labelsize=style_config['tick_fontsize'])

    # ========== 添加右侧独立图例 ==========
    categories = adata.obs[cluster_key].cat.categories
    color_map = dict(zip(categories, adata.uns[f"{cluster_key}_colors"]))

    legend_handles = [
        Line2D([0], [0],
              marker='o',
              color='w',
              markerfacecolor=color_map[cat],
              markersize=style_config['side_legend_fontsize'] * 0.8,
              label=cat,
              markeredgewidth=0)
        for cat in categories
    ]

    legend = fig.legend(
        handles=legend_handles,
        loc='center left',
        bbox_to_anchor=style_config['legend_position'],
        fontsize=style_config['side_legend_fontsize'],
        title=cluster_key.replace('_', ' ').title(),
        title_fontsize=style_config['side_legend_fontsize'] + 1,
        frameon=False
    )

    # ========== 布局优化 ==========
    plt.subplots_adjust(right=0.65)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_extra_artists=[legend], bbox_inches='tight')
    else:
        plt.show()
    plt.close()
#
plot_annotated_umap(
    mydata,
    cluster_key='major_celltype',  # 直接指定细胞类型列
    style_config={
        'dot_size': 12,
        'on_data_fontsize': 4,
        'side_legend_fontsize': 9,
        'tick_fontsize': 6,
        'legend_position': (1.05, 0.5),
        'label_offset': (0.5, -0.2),  # 确保默认包含此参数
        'fig_width': 10
            },
    save_path='E:/SCI/002/picture/S9/S9i.png'
)





#导入模块

import numpy as np

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

#sc.settings.set_figure_params(dpi=50, dpi_save=500, figsize=(7, 7))

sc.settings.set_figure_params(dpi=50, dpi_save=300, figsize=(7, 7))


dir_base = 'E:/SCI/002/data/BA2/'
#图像的文件名
filename = 'CytAssist_FFPE_Mouse_Brain_Rep2_tissue_image.tif'
img = imread(dir_base + filename)

#细胞坐标信息
gdf = gpd.read_file("E:/pythonproject/project002-2/area-BA2.shp")

gdf.head()



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
# 绘制并保存聚类结果
dir_base = 'E:/SCI/002/picture/S9/'
plot_clusters_and_save_image(title="Region of interest 1", gdf=gdf, img=img, adata=mydata, color_by_obs='major_celltype', output_name=dir_base+"S9h.tiff")

plot_clusters_and_save_image(title="Region of interest 1", gdf=gdf, img=img, adata=mydata, bbox=(11000,7000,19000,13000), color_by_obs='major_celltype', output_name=dir_base+"S9K-1.tiff")
plot_clusters_and_save_image(title="Region of interest 2", gdf=gdf, img=img, adata=mydata, bbox=(3000,12000,9000,18000), color_by_obs='major_celltype', output_name=dir_base+"S9K-2.tiff")
plot_clusters_and_save_image(title="Region of interest 3", gdf=gdf, img=img, adata=mydata, bbox=(7000,9000,12000,14000), color_by_obs='major_celltype', output_name=dir_base+"S9K-3.tiff")


cluster4annotation = {
    '0': 'neuron',
    '1': 'Oligodendrocyte',
    '2': 'Astrocyte',
    '3': 'neuron',
    '4': 'neuron',
    '5': 'neuron',
    '6': 'neuron',
    '7': 'Astrocyte',
    '8': 'neuron',
    '9': 'Oligodendrocyte',
    '10': 'neuron',
    '11': 'neuron',
    '12': 'Choroid plexis epithelial cell',
    '13': 'vascular and leptomeningeal cell',
    '14': 'neuron',
    '15': 'Astrocyte',
    '16': 'Oligodendrocyte'
}
mydata.obs['proportion_celltype'] = mydata.obs['clusters1'].map(cluster4annotation).astype('category')

# 定义颜色字典
celltype_colors = {
    'neuron': '#FF9999',
    'Oligodendrocyte': '#66B2FF',
    'Astrocyte': '#99FF99',
    'vascular and leptomeningeal cell': '#FF6666',
    'Choroid plexis epithelial cell': '#CC66FF'
}
# 预定义标签顺序（与颜色字典键名对应）
fixed_order = ['neuron', 'Oligodendrocyte', 'Astrocyte', 'vascular and leptomeningeal cell','Choroid plexis epithelial cell']  # :ml-citation{ref="5" data="citationList"}


# 提取细胞类型信息
celltype_counts = (mydata.obs['proportion_celltype'].value_counts()
                  .reindex(fixed_order)
                  .reset_index())
celltype_counts.columns = ['proportion_celltype', 'counts']
celltype_counts['proportion'] = celltype_counts['counts'] / celltype_counts['counts'].sum()

# 生成颜色列表（按 labels 顺序）
labels = celltype_counts['proportion_celltype'].tolist()
colors = [celltype_colors[label] for label in labels]

plt.figure(figsize=(10, 10))
wedges, texts, autotexts = plt.pie(
    celltype_counts['proportion'],
    labels=celltype_counts['proportion_celltype'],
    radius=0.6,
    autopct='%1.1f%%',
    startangle=90,
    colors=colors,
    pctdistance=1.1,
    labeldistance=1.35,
    wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
    textprops={'fontsize': 14}
)

for text, autotext in zip(texts, autotexts):
    label = text.get_text()
    pct_value = autotext.get_text().replace('%', '')
    text.set_text(f"{label} ({pct_value}%)")
    autotext.set_visible(False)

radius = 0.6
for i, (wedge, text) in enumerate(zip(wedges, texts)):
    theta = np.deg2rad((wedge.theta1 + wedge.theta2) / 2)

    x_start = np.cos(theta) * radius
    y_start = np.sin(theta) * radius

    if i in [3, 5]:
        x_multiplier = 2 if i == 3 else -15  # 左右分列
        y_multiplier = 1.1
    else:
        x_multiplier = 1.2 * np.sign(x_start)
        y_multiplier = 1.2

    x_text = x_multiplier * abs(x_start)
    y_text = y_multiplier * y_start

    ha = 'right' if x_start < 0 else 'left'

    plt.annotate(
        text.get_text(),
        xy=(x_start, y_start),
        xytext=(x_text, y_text),
        arrowprops=dict(
            arrowstyle="->",
            color=wedge.get_facecolor(),
            connectionstyle=f"angle,angleA=0,angleB={np.rad2deg(theta)}"
        ),
        ha=ha,
        va='center',
        fontsize=14,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=wedge.get_facecolor(), lw=1.7)
    )
    text.set_visible(False)

#plt.legend(loc="upper right", fontsize=14, bbox_to_anchor=(1.1, 1.05), borderaxespad=0.3) 图例不要了
plt.title('Proportion of Cell Subpopulations (Predicted 2)',
          pad=43,
          fontsize=15,
          fontweight='bold',
          color='#000000')
plt.subplots_adjust(top=0.85)
plt.axis('equal')
plt.savefig(dir_base + 'S9l-BA2.png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()
