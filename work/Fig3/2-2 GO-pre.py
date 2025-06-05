'''
pip install mygene goatools -i https://pypi.tuna.tsinghua.edu.cn/simple --timeout=600
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple mygene --timeout=600
'''


from goatools import obo_parser
from goatools.associations import read_gaf
from goatools.go_enrichment import GOEnrichmentStudy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mygene import MyGeneInfo
mg = MyGeneInfo()  # ← 关键初始化语句


'''''''''
from goatools.base import download_go_basic_obo
# 定义明确的下载路径（包含文件名）
obo_path = r"E:\pythonproject\project002-2\Fig3\go-basic.obo"

# 下载到指定路径
download_go_basic_obo(obo_path)  # 直接传递完整文件路径
'''''''''


# 初始化全局设置
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#333F4B'
plt.rcParams['axes.linewidth'] = 0.8




# 数据准备
godag = obo_parser.GODag("go-basic.obo")
target_genes = pd.read_csv('E:/SCI/002/picture/S8/only GAC-pre.csv', usecols=['names'])[
    'names'].str.upper().unique().tolist()

mg = MyGeneInfo()
converted = mg.querymany(target_genes, scopes='symbol', fields='uniprot.Swiss-Prot', species='human', size=1)
target_genes = [
    swissprot_id
    for hit in converted if 'uniprot' in hit
    for swissprot_id in (
        [hit['uniprot']['Swiss-Prot']] if isinstance(hit['uniprot']['Swiss-Prot'], str)
        else hit['uniprot']['Swiss-Prot']
    )
]

namespaces = {
    'BP': {'color': '#2f9fc9', 'file_suffix': 'BP'},
    'MF': {'color': '#f6b57b', 'file_suffix': 'MF'},
    'CC': {'color': '#e54d4c', 'file_suffix': 'CC'}
}







def run_go_enrichment(namespace, config):
    geneid2gos = read_gaf("goa_human.gaf", namespace=namespace, godag=godag)
    background_genes = list(geneid2gos.keys())
    valid_genes = list(set(target_genes) & set(geneid2gos.keys()))

    goea = GOEnrichmentStudy(
        background_genes,
        geneid2gos,
        godag,
        methods=['fdr_bh'],
        alpha=0.3,
        propagate_counts=True
    )
    results = goea.run_study(valid_genes)
    significant_results = [r for r in results if r.p_fdr_bh < 0.2]
    sorted_results = sorted(significant_results, key=lambda x: x.p_uncorrected)[:10]

    return {
        'namespace': namespace,
        'data': sorted_results,
        'df': pd.DataFrame([{
            'GO_ID': r.GO,
            'Term': r.name,
            '-log10(p)': -np.log10(r.p_uncorrected),
            'FDR': r.p_fdr_bh,
            'ypos': idx  # 新增位置标识
        } for idx, r in enumerate(sorted_results)])
    }


def plot_combined_results(results):
    # 合并数据并保持命名空间顺序
    y_offset = 0
    group_gap = 1.5  # 组间间距
    all_terms = []
    colors = []
    combined_df = pd.DataFrame()

    for res in results:
        df = res['df']
        if not df.empty:
            df['ypos'] = df['ypos'] + y_offset
            combined_df = pd.concat([combined_df, df], ignore_index=True)
            all_terms.extend(df['Term'].tolist())
            colors.extend([namespaces[res['namespace']]['color']] * len(df))
            y_offset += len(df) + group_gap

    # 创建画布
    fig, ax = plt.subplots(figsize=(8, 8))

    # 绘制条形图（保持分组间距）
    bars = ax.barh(
        y=combined_df['ypos'],
        width=combined_df['-log10(p)'],
        color=colors,
        edgecolor='white',
        height=0.7
    )

    # 设置y轴标签和刻度
    ax.set_yticks(combined_df['ypos'])
    ax.set_yticklabels(combined_df['Term'])
    ax.invert_yaxis()  # 重要：保持从上到下的顺序

    # 样式优化
    ax.set_title('GO Enrichment Analysis (prediction)', fontsize=14, pad=20, fontweight='bold', color='#333F4B')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('-log10(p-value)', fontsize=12, color='#333F4B', labelpad=10)
    ax.grid(axis='x', linestyle='--', alpha=0.3)

    # 添加图例
    legend_labels = [plt.Rectangle((0, 0), 1, 1, color=namespaces[ns]['color']) for ns in namespaces]
    ax.legend(legend_labels, namespaces.keys(),
              loc='lower right',
              frameon=False,
              title='Namespace',
              title_fontsize=10)

    plt.tight_layout()
    plt.savefig('grouped_go_analysis_pre_s.png', dpi=300, bbox_inches='tight')
    plt.close()


# 主执行流程
analysis_results = []
for ns, config in namespaces.items():
    result = run_go_enrichment(ns, config)
    analysis_results.append(result)
    result['df'].to_csv(f'go_enrichment_pre_{config["file_suffix"]}-s.csv', index=False)

plot_combined_results(analysis_results)


