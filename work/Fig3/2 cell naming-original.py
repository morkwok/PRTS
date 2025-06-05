import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import geopandas as gpd
from tifffile import imread, imwrite
from shapely.geometry import Polygon, Point
from matplotlib.colors import ListedColormap
sc.settings.set_figure_params(dpi=50, dpi_save=300, figsize=(7, 7))

dir_base = 'E:/pythonproject/project002-2/Fig3/'

###汇总Visium HD基因表达数据
# Load Visium HD data 加载Visium HD数据
raw_h5_file = dir_base+'Fig3 run（real）.h5ad'
mydata = sc.read_h5ad(raw_h5_file)

dir_base = 'E:/SCI/002/picture/S8/'


cluster3annotation = {
    '0': 'AC_01',
    '1': 'Neu_01',
    '2': 'OLG_01',
    '3': 'Neu_02',
    '4': 'Neu_03',
    '5': 'Neu_04',
    '6': 'Neu_05',
    '7': 'Neu_06',
    '8': 'OLG_02',
    '9': 'Neu_07',
    '10': 'Neu_08',
    '11': 'AC_02',
    '12': 'GAC',
    '13': 'Neu_09',
    '14': 'AC_03',
    '15': 'AC_04',
    '16': 'Neu_10',
    '17': 'Neu_11',
    '18': 'VLM',
    '19': 'AC_05',
    '20': 'Neu_12',
    '21': 'Neu_13'
}
mydata.obs['major_celltype'] = mydata.obs['clusters1'].map(cluster3annotation).astype('category')

#print(mydata.obs)

#S8b
marker_genes_dict = {
    'Neu': ['Olfm1','Snap25','Ywhah','Penk','Ppp1r1b','Cpe' ,'Fth1','Slc17a7' ,'Ncdn','Adcy1','Pvalb','Uchl1' ,'Scn4b','Gpr88','mt-Atp8','mt-Co1','Dnm1','Syp','Camk2a'],#神经细胞
    'OLG': ['Plp1','Cldn11','Mbp','Mobp'],#少突胶质细胞
    'AC': ['Apoe', 'Sparcl1','Aldoc','Sparc','Selenop','Mt1','Mt2','Clu'],#星形胶质细胞
    'VLM': ['Ptgds','Igf2']# 脑膜细胞
}


sc.pl.dotplot(mydata, marker_genes_dict, 'major_celltype', standard_scale="var",dendrogram=True)



custom_order = ['Neu_01','Neu_02','Neu_03','Neu_04','Neu_05','Neu_06','Neu_07','Neu_08','Neu_09','Neu_10','Neu_11','Neu_12',
                'Neu_13', 'AC_01', 'AC_02', 'AC_03', 'AC_04', 'AC_05', 'OLG_01', 'OLG_02', 'GAC','VLM']
mydata.obs['groupdotplot'] = pd.Categorical(
    mydata.obs['major_celltype'],
    categories=custom_order,
    ordered=True
)
sc.pl.dotplot(mydata, marker_genes_dict, 'groupdotplot',
             standard_scale="var", dendrogram=False)



# 导入以下模块
from matplotlib.lines import Line2D
import scanpy as sc


def plot_annotated_umap(
        adata,
        cluster_key='cluster',
        style_config=None,  # 修改为 None 并合并默认配置
        save_path=None
):
    # 合并默认配置和用户传入的配置
    default_style = {
        'dot_size': 12,
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
        'dot_size': 3,
        'on_data_fontsize': 4,
        'side_legend_fontsize': 9,
        'tick_fontsize': 6,
        'legend_position': (1.05, 0.5),
        'label_offset': (0.5, -0.2),  # 确保默认包含此参数
        'fig_width': 10
            },
    save_path='E:/SCI/002/picture/S8/S8a.png'
)

sc.tl.rank_genes_groups(mydata, groupby='major_celltype', method='wilcoxon')
sc.pl.rank_genes_groups(mydata, n_genes=25, sharey=False)

rank_results = mydata.uns['rank_genes_groups']

print(rank_results)

all_groups_results = pd.DataFrame()
group_labels = mydata.obs[rank_results['params']['groupby']].unique()
for group_label in group_labels:
    group_df = sc.get.rank_genes_groups_df(mydata, group=group_label)
    group_df = group_df.sort_values(by="scores", ascending=False)
    group_df = group_df[
        (abs(group_df['logfoldchanges']) > 1) &
        (group_df['pvals_adj'] < 0.05)  # 假设列名为pvals_adj
        ]
    group_df['group'] = group_label
    all_groups_results = pd.concat([all_groups_results, group_df], ignore_index=True)

all_groups_results.to_csv(dir_base+'Frozen-real-GO(S-table2).csv', index=False)





dir_base = 'E:/SCI/002/data/brainfixfrozen/Visium_HD_Mouse_Brain_Fixed_Frozen_binned_outputs/binned_outputs/square_002um/'
#图像的文件名
filename = 'Visium_HD_Mouse_Brain_Fixed_Frozen_tissue_image.btf'
img = imread(dir_base + filename)

#细胞坐标信息

gdf = gpd.read_file("E:/pythonproject/project002-2/area-Frozen.shp")

gdf.head()


def plot_clusters_and_save_image(title, gdf, img, adata, bbox=None, color_by_obs=None, output_name=None):
    # 自定义颜色列表
    color_list = ["#7f0000", "#808000", "#483d8b", "#008000", "#bc8f8f", "#008b8b", "#4682b4", "#000080", "#d2691e",
                  "#9acd32", "#8fbc8f", "#800080", "#b03060", "#ff4500", "#ffa500", "#ffff00", "#00ff00", "#8a2be2",
                  "#00ff7f", "#dc143c", "#00ffff", "#0000ff", "#ff00ff", "#1e90ff", "#f0e68c", "#90ee90", "#add8e6",
                  "#ff1493", "#7b68ee", "#ee82ee"]

    # 创建画布和子图 (增大画布宽度)
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))  # 宽度增加到20英寸
    plt.subplots_adjust(wspace=0.3, right=0.72)  # 调整右侧留空区域

    # 处理图像裁剪
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

    # 准备聚类数据
    merged_gdf = gdf.merge(
        adata.obs[[color_by_obs]],
        left_on='id',
        right_index=True
    )

    # 筛选位于bbox内的多边形
    if bbox_poly is not None:
        intersects = merged_gdf.geometry.intersects(bbox_poly)
        filtered_gdf = merged_gdf[intersects].copy()
    else:
        filtered_gdf = merged_gdf.copy()

    # 创建颜色映射
    categories = filtered_gdf[color_by_obs].unique()
    n_cats = len(categories)
    cmap = ListedColormap(color_list[:n_cats], name='custom_cmap')

    # 绘制聚类结果（关键参数设置）
    plot = filtered_gdf.plot(
        column=color_by_obs,
        cmap=cmap,
        ax=axes[1],
        legend=True,
        legend_kwds={
            'bbox_to_anchor': (1.28, 0.5),  # 水平位置 | 垂直位置（0=底部，1=顶部）
            'loc': 'center left',  # 锚点定位
            'frameon': False,  # 去除图例边框
            'ncol': 1,  # 单列显示
            'fontsize': 10,  # 调整字体大小
            'title': color_by_obs,  # 图例标题
            'borderpad': 0.8,  # 边距
            'labelspacing': 1.2  # 标签间距
        }
    )

    # 设置子图标题和样式
    axes[1].set_title(f"{title}\nClustering Result", fontsize=12)
    axes[1].axis('off')

    # 强制刷新布局（关键步骤）
    fig.canvas.draw()

    # 保存图片（优化保存参数）
    if output_name:
        plt.savefig(
            output_name,
            bbox_inches='tight',
            dpi=300,
            pad_inches=0.8,  # 增加边缘留白
            facecolor='white'  # 设置背景为白色
        )
    plt.close()



# 绘制并保存聚类结果
dir_base = 'E:/SCI/002/picture/Fig3/'
plot_clusters_and_save_image(title="Complete histological image-real", gdf=gdf, img=img, adata=mydata, color_by_obs='major_celltype', output_name=dir_base+"F3d.tiff")



import seaborn as sns

# ===========条形图============
# 提取细胞类型信息
celltype_counts = mydata.obs['major_celltype'].value_counts().reset_index()
celltype_counts.columns = ['major_celltype', 'counts']

# 计算比例
total_cells = celltype_counts['counts'].sum()
celltype_counts['proportion'] = celltype_counts['counts'] / total_cells

print(celltype_counts)

# 假设你的分类顺序与颜色列表顺序一致
categories = sorted(celltype_counts['major_celltype'].unique())
colors = ["#f25d9c","#ff7171", "#ee802e", "#db8f10", "#c69716",
          "#9bb222", "#6cb905", "#28b61c", "#39bc45", "#5ccd93",
          "#54d0b9", "#5fcdcf", "#5acadc", "#61badb", "#63a5d5",
          "#909bd9", "#b690db", "#ca83d5", "#d373c0", "#dd68b1",
          "#e75fa5", "#f9709e", "#979797"]

# 创建颜色字典（确保长度匹配）
color_mapping = dict(zip(categories, colors[:len(categories)]))

plt.figure(figsize=(12,6))
sns.barplot(
    x='major_celltype',
    y='proportion',
    data=celltype_counts,
    palette=color_mapping,  # 直接传入字典
    order=categories  # 强制分类顺序与颜色对应
)

plt.xticks(rotation=45, ha='right')  # 旋转x轴标签
plt.title('Proportion of Cell Subpopulations')
plt.xlabel('Cell Type')
plt.ylabel('Proportion')
plt.tight_layout()
plt.savefig(dir_base+'F3i-real.png', dpi=300, bbox_inches='tight')
plt.close()
# ===========条形图结束============




#F3I-real
# 基础绘图模块
import matplotlib.pyplot as plt
# 空间数据处理（如需）
from shapely.geometry import Polygon


cluster4annotation = {
    '0': 'Astrocyte',
    '1': 'neuron',
    '2': 'Oligodendrocyte',
    '3': 'neuron',
    '4': 'neuron',
    '5': 'neuron',
    '6': 'neuron',
    '7': 'neuron',
    '8': 'Oligodendrocyte',
    '9': 'neuron',
    '10': 'neuron',
    '11': 'Astrocyte',
    '12': 'glutamatergic astrocyte',
    '13': 'neuron',
    '14': 'Astrocyte',
    '15': 'Astrocyte',
    '16': 'neuron',
    '17': 'neuron',
    '18': 'vascular and leptomeningeal cell',
    '19': 'Astrocyte',
    '20': 'neuron',
    '21': 'neuron'
}
mydata.obs['proportion_celltype'] = mydata.obs['clusters1'].map(cluster4annotation).astype('category')

# 定义颜色字典
celltype_colors = {
    'neuron': '#2f9fc9',
    'Oligodendrocyte': '#f6b57b',
    'Astrocyte': '#e54d4c',
    'glutamatergic astrocyte': '#63bbb0',
    'vascular and leptomeningeal cell': '#9c4e8e'
}

# 预定义标签顺序（与颜色字典键名对应）
fixed_order = ['neuron', 'Oligodendrocyte', 'Astrocyte', 'glutamatergic astrocyte', 'vascular and leptomeningeal cell']

# 提取细胞类型信息
celltype_counts = (mydata.obs['proportion_celltype'].value_counts()
                  .reindex(fixed_order)
                  .reset_index())
celltype_counts.columns = ['proportion_celltype', 'counts']
celltype_counts['proportion'] = celltype_counts['counts'] / celltype_counts['counts'].sum()

# 生成颜色列表（按 labels 顺序）
labels = celltype_counts['proportion_celltype'].tolist()
colors = [celltype_colors[label] for label in labels]

from matplotlib.patches import ConnectionPatch
import numpy as np
from matplotlib.patches import ConnectionPatch  # 关键导入


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

    if i in [3, 4]:
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
plt.title('Proportion of Cell Subpopulations (real)',
          pad=43,
          fontsize=15,
          fontweight='bold',
          color='#000000')
plt.subplots_adjust(top=0.85)
plt.axis('equal')
plt.savefig(dir_base + 'F3j-real.png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()