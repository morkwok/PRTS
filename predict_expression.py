import sys
import os
import argparse
import pickle

import numpy as np
import pandas as pd
import anndata as ad
import torch
import torch.nn.functional as F
import scipy.sparse as sp

def denormalize_expression(y_pred_normalized, y_min, y_range):

    y_denormalized = np.copy(y_pred_normalized)
    for i in range(y_denormalized.shape[0]):
        for j in range(y_denormalized.shape[1]):
            if y_denormalized[i, j] > 0:
                y_denormalized[i, j] = y_denormalized[i, j] * y_range[j] + y_min[j]
    return y_denormalized


def load_model(model_path, expected_genes, gene_list_file):
    print(f"加载模型: {model_path}")
    
    if gene_list_file.endswith('.csv'):
        gene_names = pd.read_csv(gene_list_file, header=None)[0].tolist()
    else:
        with open(gene_list_file, 'r') as f:
            gene_names = [line.strip() for line in f]

    if model_path.endswith(('.pt', '.pth')):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(model_path, map_location=device)
        
        model = CustomPytorchRegressor(
            input_size=checkpoint['config']['input_size'],
            output_size=checkpoint['config']['output_size'],
            hidden_layer_sizes=checkpoint['config']['hidden_layer_sizes'],
        )
        model.model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        
        if checkpoint['config']['output_size'] != expected_genes:
            raise ValueError(f"模型输出维度({checkpoint['config']['output_size']})与基因数({expected_genes})不匹配")
        
        return model, gene_names

    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    if isinstance(model_data, dict):
        if 'torch_model_path' in model_data:
            config = model_data['model_config']
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            model = CustomPytorchRegressor(
                input_size=config['input_size'],
                output_size=config['output_size'],
                hidden_layer_sizes=config['hidden_layer_sizes'],
                alpha=config['alpha'],
                batch_size=config['batch_size'],
                binary_weight=config['binary_weight'],
                expression_weight=config['expression_weight'],
                device=device
            )
            model.model = model._init_model()
            torch_state = torch.load(model_data['torch_model_path'], map_location=device)
            model.model.load_state_dict(torch_state['state_dict'])
            model.model.eval()
            
            return model, gene_names

        return model_data.get('model'), gene_names
    
    return model_data, gene_names

def load_features(feature_file):

    print(f"加载特征文件: {feature_file}")
    with open(feature_file, 'rb') as f:
        data = pickle.load(f)
    
    # 错误处理1: 检查必要字段
    required_keys = ['cell_ids', 'features']
    for key in required_keys:
        if key not in data:
            raise KeyError(f"特征文件缺少必要字段: {key}")
    
    # 错误处理2: 处理不同特征格式
    features = data['features']
    if isinstance(features, np.ndarray):
        cls_features = features
    elif isinstance(features, dict):
        if 'cls' not in features:
            raise KeyError("features字典中缺少'cls'键")
        cls_features = features['cls']
    else:
        raise TypeError(f"不支持的features类型: {type(features)}")
    
    print(f"特征矩阵形状: {cls_features.shape}")
    return cls_features, data['cell_ids']

def load_train_data(h5ad_file, gene_list_file=None):

    print(f"加载训练数据: {h5ad_file}")
    adata = ad.read_h5ad(h5ad_file)
    
    if gene_list_file is not None:
        if gene_list_file.endswith('.csv'):
            gene_names = pd.read_csv(gene_list_file, header=None)[0].tolist()
        else:
            with open(gene_list_file, 'r') as f:
                gene_names = [line.strip() for line in f]

        gene_mask = adata.var_names.isin(gene_names)
        adata = adata[:, gene_mask]
        

    if hasattr(adata.X, "toarray"):
        y_train = adata.X.toarray()
    else:
        y_train = adata.X
    
    print(f"训练数据矩阵形状: {y_train.shape}")
    return y_train



def predict_mlp(model, X, y_train):
    print("使用MLP模型进行预测...")

    if np.any(~np.isfinite(X)):
        print("发现 NaN 或无穷值，替换为 0")
        X = np.nan_to_num(X)

    if hasattr(model, 'predict'):
        y_pred_scaled = model.predict(X)
    else:
        raise RuntimeError("模型未实现 predict 接口")
    
    print(f"模型:{y_pred_scaled.shape}, 模型输出维度: {y_pred_scaled.shape[1]}, 训练数据基因数: {y_train.shape[1]}")

    with open(r'train_256\embedd\normalization_info.pkl', 'rb') as f:
        normalization_info = pickle.load(f)

    y_min = normalization_info['y_min']
    y_range = normalization_info['y_range']

    # y_pred_normalized: 预测得到的归一化表达量，shape = [N, G]

    y_pred = denormalize_expression(y_pred_scaled, y_min, y_range)   

    y_pred[y_pred < 0.5] = 0.0
    print(f"预测非零比例: {(y_pred > 0).sum() / y_pred.size * 100:.2f}%")
    return y_pred
    #return y_pred_scaled


def save_predictions(y_pred, cell_ids, gene_names, output_file, format='h5ad'):
    """保存预测结果到文件"""
    print(f"保存预测结果到: {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"非零值数量: {np.sum(y_pred > 0)}")
    print(f"最小非零值: {np.min(y_pred[y_pred > 0])}")
    print(f"最大值: {np.max(y_pred)}")

    if not gene_names:
        print("警告：基因名称列表为空或长度与预测结果列数不匹配，尝试生成默认基因名称。")
        gene_names = [f"Gene_{i}" for i in range(y_pred.shape[1])]

    if format == 'h5ad':
        sparse_y = sp.csr_matrix(y_pred)
        adata = ad.AnnData(
            X=sparse_y, 
            obs=pd.DataFrame(index=cell_ids), 
            var=pd.DataFrame(index=gene_names)
        )
        adata.write(output_file)
    else:
        pd.DataFrame(y_pred, index=cell_ids, columns=gene_names).to_csv(output_file)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="预测细胞基因表达量")
    parser.add_argument("--model", required=True, help="模型文件路径")
    parser.add_argument("--feature_file", required=True, help="特征文件路径")
    parser.add_argument("--train_data", required=True, help="训练数据文件路径") 
    parser.add_argument("--gene_list", required=True) # 新增参数
    parser.add_argument("--output_file", required=True, help="输出文件路径")
    parser.add_argument("--output_format", choices=['h5ad', 'csv'], default='h5ad')
    return parser.parse_args()

def main():
    args = parse_args()

    y_train = load_train_data(
        h5ad_file=args.train_data, 
        gene_list_file=args.gene_list  
    )
    n_genes = y_train.shape[1]
    model, gene_names = load_model(args.model, expected_genes=n_genes, gene_list_file=args.gene_list)

    X, cell_ids = load_features(args.feature_file)
    y_pred = predict_mlp(model, X, y_train)
    
    if y_pred is not None:
        save_predictions(y_pred, cell_ids, gene_names, args.output_file, args.output_format)

if __name__ == '__main__':
    main()