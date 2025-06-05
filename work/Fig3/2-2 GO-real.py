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
target_genes = pd.read_csv('E:/SCI/002/picture/S8/only GAC-real.csv', usecols=['names'])[
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
    ax.set_title('GO Enrichment Analysis (Actual)', fontsize=14, pad=20, fontweight='bold', color='#333F4B')
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
    plt.savefig('grouped_go_analysis_real_s.png', dpi=300, bbox_inches='tight')
    plt.close()


# 主执行流程
analysis_results = []
for ns, config in namespaces.items():
    result = run_go_enrichment(ns, config)
    analysis_results.append(result)
    result['df'].to_csv(f'go_enrichment_real_{config["file_suffix"]}-s.csv', index=False)

plot_combined_results(analysis_results)

















'''
# 数据准备
godag = obo_parser.GODag("go-basic.obo")
target_genes = pd.read_csv('E:/SCI/002/picture/S8/only GAC-real.csv', usecols=['names'])['names'].tolist()  # 明确提取列数据

# 基因符号转Entrez ID（添加错误处理）
converted = mg.querymany(
    target_genes,
    scopes='symbol',
    fields='entrezgene',
    species='mouse',
    size=1,
    as_dataframe=True
)
target_entrez = converted[~converted['entrezgene'].isna()]['entrezgene'].astype(str).tolist()

# 更新命名空间颜色配置（可选）
namespaces = {
    'BP': {'color': '#4e79a7', 'file_suffix': 'BP'},  # 调整颜色
    'MF': {'color': '#f28e2b', 'file_suffix': 'MF'},
    'CC': {'color': '#e15759', 'file_suffix': 'CC'}
}

# 配置参数
config = {
    'species': 'mouse',  # 指定物种
    'gaf_path': {
        'mouse': 'mgi.gaf',
        'human': 'goa_human.gaf'
    },
    'alpha': 0.3,
    'propagate_counts': True
}


def run_go_enrichment(namespace, config):
    """通用富集分析函数（支持鼠/人）"""
    # 动态加载对应物种的GAF
    gaf_file = config['gaf_path'][config['species']]
    geneid2gos = read_gaf(gaf_file, namespace=namespace, godag=godag)
    background_genes = [str(g) for g in geneid2gos.keys()]

    # 获取有效基因（Entrez ID）
    valid_genes = list(set(target_entrez) & set(background_genes))
    if not valid_genes:
        return {'namespace': namespace, 'df': pd.DataFrame()}

    # 初始化GOEA对象
    goea = GOEnrichmentStudy(
        background_genes,
        geneid2gos,
        godag,
        methods=['fdr_bh'],
        alpha=config['alpha'],
        propagate_counts=config['propagate_counts']
    )

    # 运行分析并过滤结果
    results = goea.run_study(valid_genes)
    significant_results = [r for r in results if r.p_fdr_bh < 0.2]
    sorted_results = sorted(significant_results, key=lambda x: x.p_uncorrected)[:10]

    # 构造结果DataFrame
    return {
        'namespace': namespace,
        'data': sorted_results,
        'df': pd.DataFrame([{
            'GO_ID': r.GO,
            'Term': r.name,
            '-log10(p)': -np.log10(r.p_uncorrected),
            'FDR': r.p_fdr_bh,
            'ypos': idx
        } for idx, r in enumerate(sorted_results)])
    } if sorted_results else {'namespace': namespace, 'df': pd.DataFrame()}



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
    ax.set_title('GO Enrichment Analysis (Actual)', fontsize=14, pad=20, fontweight='bold', color='#333F4B')
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
    plt.savefig('grouped_go_analysis_real_s.png', dpi=300, bbox_inches='tight')
    plt.close()



# 主执行流程修复
analysis_results = []
global_config = {  # 定义全局配置字典
    'species': 'mouse',
    'gaf_path': {'mouse': 'mgi.gaf', 'human': 'goa_human.gaf'},
    'alpha': 0.3,
    'propagate_counts': True,
    'namespaces': namespaces  # 添加命名空间配置
}

for ns_key in namespaces.keys():  # 明确遍历命名空间
    try:
        print(f"\n=== 开始分析命名空间 {ns_key} ===")

        # 验证基因列表有效性
        if not target_entrez:
            raise ValueError("无有效Entrez ID，请检查基因符号映射")

        # 执行富集分析
        result = run_go_enrichment(ns_key, global_config)  # 传入正确的config

        # 结果有效性检查
        if result['df'].empty:
            print(f"命名空间 {ns_key} 无显著结果")
        else:
            print(f"发现 {len(result['df'])} 个显著条目")

        analysis_results.append(result)
        result['df'].to_csv(f'go_enrichment_real_{namespaces[ns_key]["file_suffix"]}-s.csv', index=False)

    except Exception as e:
        print(f"命名空间 {ns_key} 分析失败: {str(e)}")
        analysis_results.append({'namespace': ns_key, 'df': pd.DataFrame()})

# 合并可视化
plot_combined_results(analysis_results)
'''''''''''