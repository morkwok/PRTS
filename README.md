# iSTAR 细胞基因表达预测工具

本项目提供了一套用于从细胞图像预测基因表达量的工具。基于HIPT（分层图像金字塔Transformer）模型，可以从细胞图像中提取特征，然后使用这些特征来预测细胞的基因表达谱。

## 功能特点

- 从细胞图像中提取多尺度特征（全局和局部）
- 支持批处理大规模细胞图像数据，避免内存溢出
- 提供两种基因表达量预测模型：
  - 多层感知机 (MLP) 回归模型
  - 基于最近邻的回归模型
- 支持预测结果的可视化和评估
- 完整的数据处理流程，从图像到基因表达预测

## 主要组件

1. **特征提取 (`extract_features.py`)**：
   - 从细胞图像中提取多尺度特征
   - 支持批处理大规模数据
   - 合并局部和全局特征

2. **特征合并 (`merge_features.py`)**：
   - 合并多个批次处理的特征

3. **模型训练 (`train_model.py`)**：
   - 训练MLP或最近邻模型
   - 管理特征标准化
   - 自动保存训练数据和模型参数

4. **基因表达预测 (`predict_expression.py`)**：
   - 使用训练好的模型预测基因表达量
   - 支持预测方差（针对最近邻模型）
   - 提供结果可视化

## 安装要求

```
numpy
pandas
anndata
scikit-learn
matplotlib
torch
```

## 使用方法

### 1. 特征提取

从细胞图像中提取特征：

```bash
python extract_features.py \
    --prefix /path/to/images/ \
    --h5ad-file /path/to/cells.h5ad \
    --device cuda \
    --batch-size 100
```

如使用批处理，合并特征：

```bash
python merge_features.py \
    --input-dir /path/to/embedd \
    --output-file /path/to/cell_features.pkl
```

### 2. 模型训练

训练MLP模型：

```bash
python train_model.py \
    --feature_file /path/to/cell_features.pkl \
    --h5ad_file /path/to/gene_expression.h5ad \
    --gene_list /path/to/gene_list.txt \
    --output_model /path/to/mlp_model.pkl \
    --model_type mlp \
    --hidden_layers "256,256,256,128" \
    --alpha 0.01
```

训练最近邻模型：

```bash
python train_model.py \
    --feature_file /path/to/cell_features.pkl \
    --h5ad_file /path/to/gene_expression.h5ad \
    --gene_list /path/to/gene_list.txt \
    --output_model /path/to/nn_model.pkl \
    --model_type nn \
    --n_neighbors 10 \
    --weights distance
```

### 3. 基因表达预测

使用MLP模型预测：

```bash
python predict_expression.py \
    --model /path/to/mlp_model.pkl \
    --feature_file /path/to/new_cell_features.pkl \
    --output_file /path/to/predictions.h5ad \
    --visualize
```

使用最近邻模型预测：

```bash
python predict_expression.py \
    --model /path/to/nn_model.pkl \
    --feature_file /path/to/new_cell_features.pkl \
    --train_data /path/to/nn_model_train_data.pkl \
    --output_file /path/to/predictions.h5ad \
    --save_variance \
    --visualize
```

## 示例脚本

查看 `example_usage.sh` 获取完整的使用示例。

## 数据输出格式

- 特征文件：pickle格式，包含`cell_ids`和`features`字典
- 模型文件：pickle格式，包含训练好的模型和相关参数
- 预测结果：h5ad或csv格式，包含预测的基因表达量
- 可视化结果：保存为PNG图像文件

## 注意事项

- 对于最近邻模型，需要保存和提供原始训练数据
- 特征提取是计算密集型任务，建议在GPU上运行
- 处理大规模数据时，使用批处理功能以避免内存溢出 
