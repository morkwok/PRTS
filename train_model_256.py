import os
import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, r2_score, mean_squared_error
import anndata as ad
import copy
import argparse

class DualOutputMLP(nn.Module):
    def __init__(self, input_size=384, output_size=1820, hidden_sizes=(512, 512, 1024, 1024)):
        super(DualOutputMLP, self).__init__()
        layers = []
        prev_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.BatchNorm1d(size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_size = size

        self.fc_layers = nn.Sequential(*layers)
        self.output_expression = nn.Linear(prev_size, output_size)
        self.output_binary     = nn.Linear(prev_size, output_size)

    def forward(self, x):
        x = self.fc_layers(x)
        expr_pred = self.output_expression(x)
        bin_pred  = self.output_binary(x)
        return expr_pred, bin_pred

class FocalLoss(nn.Module):
    """Binary focal loss"""
    def __init__(self, alpha=0.7, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        prob = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce * (1 - p_t) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * loss
        return loss.mean() if self.reduction=='mean' else loss.sum()

def masked_mse(pred, target, mask):
    """
    只对 mask==1 的位置计算 MSE，mask.shape == pred.shape
    """
    diff = (pred - target) * mask
    return (diff.pow(2).sum() / (mask.sum() + 1e-6))

class CustomPytorchRegressor:
    def __init__(self, input_size=384, output_size=1820, hidden_layer_sizes=(512,512,1024,1024),
                 alpha=1e-3, batch_size=512, max_iter=50,
                 lr=5e-4, binary_weight=1.0, expression_weight=30.0, device=None):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.lr = lr
        self.binary_weight = binary_weight
        self.expression_weight = expression_weight
        self.model = None
        self.device = torch.device('cuda' if (device is None and torch.cuda.is_available()) else 'cpu') \
                      if device is None else device

    def _init_model(self):
        return DualOutputMLP(
            input_size=self.input_size,
            output_size=self.output_size,
            hidden_sizes=self.hidden_layer_sizes
        ).to(self.device)

    def fit(self, X_train, y_train_combined, X_val, y_val_combined):
        # 转张量
        y_train_expr, y_train_bin = y_train_combined
        X_tr = torch.FloatTensor(X_train).to(self.device)
        Y_tr_e = torch.FloatTensor(y_train_expr).to(self.device)
        Y_tr_b = torch.FloatTensor(y_train_bin).to(self.device)

        y_val_expr, y_val_bin = y_val_combined
        X_v = torch.FloatTensor(X_val).to(self.device)
        Y_v_e = torch.FloatTensor(y_val_expr).to(self.device)
        Y_v_b = torch.FloatTensor(y_val_bin).to(self.device)

        # 计算 pos_weight（可选保留）
        pos_w = ( (y_train_bin.shape[0] - y_train_bin.sum(axis=0)) /
                  (y_train_bin.sum(axis=0) + 1e-6) ).astype(np.float32)
        pos_w_tensor = torch.FloatTensor(pos_w).to(self.device)

        # 数据加载
        train_ds = TensorDataset(X_tr, Y_tr_e, Y_tr_b)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, drop_last=True)
        val_loader   = DataLoader(TensorDataset(X_v, Y_v_e, Y_v_b),
                                  batch_size=self.batch_size, shuffle=False)

        # 模型、优化器、调度
        if self.model is None:
            self.model = self._init_model()

        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.alpha)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                         factor=0.5, patience=10, verbose=True)

        # 损失函数
        focal_loss = FocalLoss(alpha=0.7, gamma=2.0)
        # 用 masked_mse 直接计算

        best_loss = float('inf')
        no_improve = 0
        history = {'train_loss':[], 'val_loss':[]}
        best_acc = 0
        best_rec = 0
        best_state = None

        for epoch in range(1, self.max_iter+1):
            # 训练
            self.model.train()
            epoch_loss = 0.0
            for Xb, Ye, Yb in train_loader:
                optimizer.zero_grad()
                pred_e, pred_b_logits = self.model(Xb)

                loss_b = focal_loss(pred_b_logits, Yb)
                loss_e = masked_mse(pred_e, Ye, Yb)

                loss = self.binary_weight*loss_b + self.expression_weight*loss_e
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()

            avg_train = epoch_loss / len(train_loader)

            # 验证
            self.model.eval()
            val_loss = 0.0
            all_bin_logits = []
            all_bin_true   = []
            all_expr_pred  = []
            all_expr_true  = []
            with torch.no_grad():
                for Xb, Ye, Yb in val_loader:
                    pred_e, pred_b_logits = self.model(Xb)
                    loss_b = focal_loss(pred_b_logits, Yb)
                    loss_e = masked_mse(pred_e, Ye, Yb)
                    val_loss += (self.binary_weight*loss_b + self.expression_weight*loss_e).item()

                    # 保存用于计算指标
                    all_bin_logits.append(pred_b_logits.cpu())
                    all_bin_true.append(Yb.cpu())
                    all_expr_pred.append(pred_e.cpu())
                    all_expr_true.append(Ye.cpu())

            avg_val = val_loss / len(val_loader)

            # 拼接所有 batch
            all_bin_logits = torch.cat(all_bin_logits, dim=0).numpy()
            all_bin_true   = torch.cat(all_bin_true,   dim=0).numpy()
            all_expr_pred  = torch.cat(all_expr_pred,  dim=0).numpy()
            all_expr_true  = torch.cat(all_expr_true,  dim=0).numpy()

            # 计算二分类概率 & 二分类预测
            bin_proba = 1 / (1 + np.exp(-all_bin_logits))
            bin_pred  = (bin_proba >= 0.3).astype(np.float32)

            # Classification metrics
            acc  = (bin_pred == all_bin_true).mean()
            pre  = precision_score(all_bin_true.ravel(), bin_pred.ravel(), zero_division=0)
            rec  = recall_score(all_bin_true.ravel(),    bin_pred.ravel(), zero_division=0)
            f1   = f1_score(all_bin_true.ravel(),        bin_pred.ravel(), zero_division=0)

            # Regression MSE only on expressed positions
            mask = all_bin_true > 0
            if mask.sum() > 0:
                mse = ((all_expr_pred[mask] - all_expr_true[mask])**2).mean()
            else:
                mse = float('nan')

            # 打印所有指标
            print(f"Epoch {epoch}/{self.max_iter} — "
                  f"train_loss: {avg_train:.4f}, val_loss: {avg_val:.4f}  "
                  f"ACC: {acc:.4f}, PREC: {pre:.4f}, REC: {rec:.4f}, F1: {f1:.4f}, MSE: {mse:.6f}")
            
            if avg_val < best_loss:
                best_loss = avg_val
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > 20:
                    print("Early stopping.")
                    break

            if acc >= 0.7 and rec > best_rec:
                best_acc = acc
                best_rec = rec
                best_state = copy.deepcopy(self.model.state_dict())

        # 保存最佳信息到模型属性
        self.best_loss = best_loss
        self.best_acc = best_acc
        self.best_rec = best_rec

        # 加载最佳模型状态
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.history = history
        return self

    def predict(self, X, threshold=0.2):
        self.model.eval()
        X_t = torch.FloatTensor(X).to(self.device)
        expr_preds, bin_logits = self.model(X_t)
        bin_proba = torch.sigmoid(bin_logits).detach().cpu().numpy() 
        expr     = expr_preds.cpu().detach().numpy()
        bin_pred = (bin_proba >= threshold).astype(np.float32)
        return expr * bin_pred

def load_data(feature_file, h5ad_file, gene_list_file=None):
    """
    加载特征和基因表达数据
    参数:
        feature_file: 包含特征的pickle文件路径
        h5ad_file: 包含基因表达数据的h5ad文件路径
        gene_list_file: 可选，包含基因名称列表的文件路径
    返回:
        X: 特征矩阵，形状为 [n_cells, n_features]
        y: 基因表达矩阵，形状为 [n_cells, n_genes]
        cell_ids: 细胞ID列表
        gene_names: 基因名称列表
    """
    print(f"加载特征文件: {feature_file}")
    # 加载特征数据
    with open(feature_file, 'rb') as f:
        data = pickle.load(f)
    
    # 提取cell_ids和特征
    cell_ids = data['cell_ids']
    features = data['features']
    
    # 修改部分：处理 features 为 numpy.ndarray 的情况
    if isinstance(features, np.ndarray):
        cls_features = features
    elif isinstance(features, dict):
        cls_features = features['cls']
    else:
        print(f"错误: features 类型不支持，实际类型为 {type(features)}")
        print(f"features 内容: {features}")
        raise ValueError("features 必须是字典类型或 numpy.ndarray 类型")
    
    X = cls_features
    print(f"特征矩阵形状: {X.shape}")
    
    # 加载基因表达数据
    print(f"加载基因表达数据: {h5ad_file}")
    adata = ad.read_h5ad(h5ad_file)
    
    # 如果提供了基因列表文件，则只使用指定的基因
    if gene_list_file is not None:
        with open(gene_list_file, 'r', encoding='utf-8-sig') as f:
            gene_names = [line.strip() for line in f]
        # 过滤基因
        gene_mask = adata.var_names.isin(gene_names)
        if sum(gene_mask) < len(gene_names):
            missing_genes = set(gene_names) - set(adata.var_names[gene_mask])
            print(f"警告: {len(missing_genes)}个基因在数据中未找到。")
            print("缺失的基因是:", missing_genes)  # 新增这行打印具体缺失基因
        # 只保留指定的基因
        adata = adata[:, gene_mask]
        gene_names = list(adata.var_names)
    else:
        # 使用所有基因
        gene_names = list(adata.var_names)
    
    # 确保细胞ID的顺序与特征矩阵匹配
    adata_cell_indices = []
    for cell_id in cell_ids:
        if cell_id in adata.obs_names:
            idx = adata.obs_names.get_loc(cell_id)
            adata_cell_indices.append(idx)
        else:
            adata_cell_indices.append(-1)
    
    # 找出有效的细胞索引（在adata中存在的细胞）
    valid_indices = [i for i, idx in enumerate(adata_cell_indices) if idx != -1]
    valid_adata_indices = [idx for idx in adata_cell_indices if idx != -1]
    
    if len(valid_indices) < len(cell_ids):
        print(f"警告: 只找到{len(valid_indices)}/{len(cell_ids)}个细胞在基因表达数据中。")
    
    # 提取有效的特征和细胞ID
    X = X[valid_indices]
    cell_ids = [cell_ids[i] for i in valid_indices]
    
    # 提取基因表达矩阵
    y = adata.X[valid_adata_indices]
    
    # 将稀疏矩阵转换为普通矩阵
    if hasattr(y, "toarray"):
        y = y.toarray()
    
    print(f"基因表达矩阵形状: {y.shape}")
    print(f"共有{len(gene_names)}个基因")
    
    return X, y, cell_ids, gene_names

def parse_args():
    parser = argparse.ArgumentParser(description="训练基因表达预测模型")
    parser.add_argument("--train_feature_file", type=str, required=True,
                        help="训练集包含细胞特征的pickle文件路径，由merge_features_256.py生成")
    parser.add_argument("--train_h5ad_file", type=str, required=True,
                        help="训练集包含基因表达数据的h5ad文件路径")
    parser.add_argument("--val_feature_file", type=str, required=True,
                        help="验证集包含细胞特征的pickle文件路径，由merge_features_256.py生成")
    parser.add_argument("--val_h5ad_file", type=str, required=True,
                        help="验证集包含基因表达数据的h5ad文件路径")
    parser.add_argument("--gene_list", type=str, default=None,
                        help="包含基因名称列表的文件路径，如果不提供则使用所有基因")
    parser.add_argument("--output_model", type=str, required=True,
                        help="输出模型的文件路径")
    parser.add_argument("--train_data_output", type=str, default=None,
                        help="训练数据输出文件路径(如果不提供则基于模型路径生成)")
    parser.add_argument("--model_type", type=str, choices=['nn', 'mlp'], default='mlp',
                        help="模型类型，'nn'为最近邻近模型，'mlp'为多层感知机")
    parser.add_argument("--hidden_layers", type=str, default="512,512,1024,1024",
                        help="MLP模型的隐藏层大小，以逗号分隔")
    parser.add_argument("--alpha", type=float, default=1e-3,
                        help="MLP模型的正则化参数")
    parser.add_argument("--max_iter", type=int, default=50,
                        help="MLP模型的最大迭代次数")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 加载训练集数据
    X_train, y_train, _, gene_names = load_data(args.train_feature_file, args.train_h5ad_file, args.gene_list)
    # 加载验证集数据
    X_val, y_val, _, _ = load_data(args.val_feature_file, args.val_h5ad_file, args.gene_list)
    
    # 处理具有高度偏斜分布的数据(大量零值)
    print("使用双输出模型分别处理'是否表达'和'表达量多少'两个子问题")
    
    # 标准化特征 - 仅使用训练集的统计信息
    x_mean = X_train.mean(0)
    x_std = X_train.std(0)
    # 防止除零
    x_std[x_std < 1e-8] = 1.0
    X_train_scaled = (X_train - x_mean) / x_std
    X_val_scaled = (X_val - x_mean) / x_std
    
    # 创建二分类标签：基因是否表达
    y_train_binary = (y_train > 0).astype(np.float32)
    y_val_binary = (y_val > 0).astype(np.float32)
    non_zero_ratio = y_train_binary.mean()
    print(f"目标矩阵非零值占比: {non_zero_ratio * 100:.2f}%")
    
    # 对于表达值，只处理非零值
    y_train_expression = np.copy(y_train)
    y_val_expression = np.copy(y_val)
    
    # 计算每个基因的非零值统计 - 仅使用训练集的统计信息
    y_min = np.zeros(y_train.shape[1], dtype=np.float32)
    y_max = np.zeros(y_train.shape[1], dtype=np.float32)
    
    for i in range(y_train.shape[1]):
        non_zero_vals = y_train[:, i][y_train[:, i] > 0]
        if len(non_zero_vals) > 0:
            y_min[i] = non_zero_vals.min()
            y_max[i] = non_zero_vals.max()
        else:
            y_min[i] = 0
            y_max[i] = 1
    
    # 防止除以0
    y_range = y_max - y_min
    y_range[y_range < 1e-12] = 1.0
    
    # 对非零表达值进行Min-Max标准化
    def normalize_expression(y, y_min, y_range):
        y_normalized = np.copy(y)
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                if y[i, j] > 0:
                    y_normalized[i, j] = (y[i, j] - y_min[j]) / y_range[j]
        return y_normalized
    
    y_train_expression = normalize_expression(y_train, y_min, y_range)
    y_val_expression = normalize_expression(y_val, y_min, y_range)
    
    print(f"训练集表达值矩阵形状: {y_train_expression.shape}, 二分类标签形状: {y_train_binary.shape}")
    print(f"验证集表达值矩阵形状: {y_val_expression.shape}, 二分类标签形状: {y_val_binary.shape}")
    
    hidden_layer_sizes = tuple(map(int, args.hidden_layers.split(',')))
    
    model = CustomPytorchRegressor(
        input_size=X_train_scaled.shape[1],
        output_size=y_train_expression.shape[1],
        hidden_layer_sizes=hidden_layer_sizes,
        alpha=args.alpha,
        max_iter=args.max_iter,
        batch_size=512,
        lr=0.001,
        binary_weight=1.0,
        expression_weight=30.0
    )
    
    # 直接使用加载的训练集和验证集数据
    model.fit(
        X_train_scaled, (y_train_expression, y_train_binary),
        X_val_scaled, (y_val_expression, y_val_binary)
    )
    
    # 输出最终保存的最佳模型的loss以及准确率等信息
    print(f"最佳模型验证损失: {model.best_loss:.4f}")
    print(f"最佳模型准确率: {model.best_acc:.4f}")
    print(f"最佳模型召回率: {model.best_rec:.4f}")
    
    # 保存基因归一化相关信息为文件
    normalization_info = {
        'x_mean': x_mean,
        'x_std': x_std,
        'y_min': y_min,
        'y_max': y_max,
        'y_range': y_range
    }
    normalization_file = os.path.join(os.path.dirname(args.output_model), 'normalization_info.pkl')
    with open(normalization_file, 'wb') as f:
        pickle.dump(normalization_info, f)
    print(f"基因归一化相关信息已保存到 {normalization_file}")
    
    # 保存模型
    with open(args.output_model, 'wb') as f:
        pickle.dump(model, f)
    print(f"模型已保存到 {args.output_model}")

if __name__ == '__main__':
    main()
