import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score, recall_score, precision_score, mean_squared_error, r2_score

def compare_matrices(file1, file2, gene_list_file, threshold=0):
    # 读取基因列表
    if gene_list_file.endswith('.csv'):
        gene_names = pd.read_csv(gene_list_file, header=None)[0].tolist()
    else:  # 假设是txt文件，每行一个基因
        with open(gene_list_file, 'r') as f:
            gene_names = [line.strip() for line in f]
    
    # 读取两个h5ad文件
    adata1 = ad.read_h5ad(file1)
    adata2 = ad.read_h5ad(file2)
    
    # 使用基因列表筛选基因
    gene_mask1 = adata1.var_names.isin(gene_names)
    adata1 = adata1[:, gene_mask1]
    gene_mask2 = adata2.var_names.isin(gene_names)
    adata2 = adata2[:, gene_mask2]
    
    # 获取X矩阵
    X1 = adata1.X
    X2 = adata2.X
    
    # 转换为密集矩阵进行计算
    X1_dense = X1.toarray() if hasattr(X1, 'toarray') else np.array(X1)
    X2_dense = X2.toarray() if hasattr(X2, 'toarray') else np.array(X2)
    
    # 应用阈值到 file2 的矩阵
    X2_dense[X2_dense < threshold] = 0
    
    # 分析 file1 矩阵
    X1_matrix_size = X1_dense.size
    X1_zero_count = np.sum(X1_dense == 0)
    X1_zero_ratio = X1_zero_count / X1_matrix_size
    X1_non_zero_values = X1_dense[X1_dense != 0]
    if X1_non_zero_values.size > 0:
        X1_max_non_zero = np.max(X1_non_zero_values)
        X1_min_non_zero = np.min(X1_non_zero_values)
    else:
        X1_max_non_zero = 0
        X1_min_non_zero = 0
    print(f"file1 中矩阵大小: {X1_matrix_size}")
    print(f"file1 中矩阵零值占比: {X1_zero_ratio * 100:.2f}%")
    print(f"file1 中矩阵最大非零值: {X1_max_non_zero}")
    print(f"file1 中矩阵最小非零值: {X1_min_non_zero}")
    
    # 分析 file2 矩阵
    matrix_size = X2_dense.size
    zero_count = np.sum(X2_dense == 0)
    zero_ratio = zero_count / matrix_size
    non_zero_values = X2_dense[X2_dense != 0]
    if non_zero_values.size > 0:
        min_non_zero = np.min(non_zero_values)
        max_non_zero = np.max(non_zero_values)
        # 计算缩放因子
        scale_factor = 1 / min_non_zero
        # 应用缩放因子，保留浮点数类型
        X2_dense = (X2_dense * scale_factor)
        # 新增舍入代码
        X2_dense = np.round(X2_dense)
    else:
        scale_factor = 1
        min_non_zero = 0
        max_non_zero = 0

    # 分析映射后的 file2 矩阵
    X2_mapped_matrix_size = X2_dense.size
    X2_mapped_zero_count = np.sum(X2_dense == 0)
    X2_mapped_non_zero_values = X2_dense[X2_dense != 0]
    if X2_mapped_non_zero_values.size > 0:
        X2_mapped_max_non_zero = np.max(X2_mapped_non_zero_values)
        X2_mapped_min_non_zero = np.min(X2_mapped_non_zero_values)
    else:
        X2_mapped_max_non_zero = 0
        X2_mapped_min_non_zero = 0

    print(f"缩放因子: {scale_factor}")
    print(f"映射后 file2 中矩阵大小: {X2_mapped_matrix_size}")
    print(f"映射后 file2 中矩阵最大非零值: {X2_mapped_max_non_zero}")
    print(f"映射后 file2 中矩阵最小非零值: {X2_mapped_min_non_zero}")
    print(f"file2 中矩阵大小: {matrix_size}")
    print(f"file2 中矩阵零值占比: {zero_ratio * 100:.2f}%")
    print(f"file2 中矩阵最大非零值: {max_non_zero}")
    print(f"file2 中矩阵最小非零值: {min_non_zero}")
    
    # 转换为二进制矩阵（非零值视为1）
    X1_binary = (X1_dense != 0).astype(int)
    X2_binary = (X2_dense != 0).astype(int)

    # 展平矩阵
    X1_flat = X1_binary.flatten()
    X2_flat = X2_binary.flatten()

    # 计算准确率、召回率和精确率
    accuracy = accuracy_score(X1_flat, X2_flat)
    recall = recall_score(X1_flat, X2_flat)
    precision = precision_score(X1_flat, X2_flat)

    # 获取非零值的索引
    non_zero_indices = np.logical_and(X1_dense != 0, X2_dense != 0)
    X1_non_zero = X1_dense[non_zero_indices]
    X2_non_zero = X2_dense[non_zero_indices]

    # 计算 MSE 和 R2
    mse = mean_squared_error(X1_non_zero, X2_non_zero)
    r2 = r2_score(X1_non_zero, X2_non_zero)

    # 输出结果
    print(f"准确率: {accuracy:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"决定系数 (R2): {r2:.4f}")

# 示例调用
file1 = "BrainHD_results-Frozen.h5ad"
file2 = r"valid_256\embedd\valid.h5ad"
gene_list_file = "genes.csv"
threshold = 0.01  # 可以根据需要调整阈值
compare_matrices(file1, file2, gene_list_file, threshold)