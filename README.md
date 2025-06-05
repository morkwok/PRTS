# PRTS细胞基因表达预测

本方法用于从细胞图像预测基因表达量。基于ViT模型，从细胞图像中提取特征，然后使用这些特征来预测细胞的基因表达谱。

## 目录结构

├── main\    // 主程序文件夹  
│   ├── preprocess_image.py    // 用于预处理图像的Python脚本  
│   ├── rescale.py    // 用于图像缩放的Python脚本  
│   ├── extract_features.py    // 用于从细胞图像中提取特征的Python脚本  
│   ├── merge_features.py    // 用于合并特征的Python脚本  
│   ├── train_model.py    // 用于训练模型的Python脚本  
│   ├── predict_expression.py    // 用于预测基因表达量的Python脚本  
│   ├── model_utils.py    // 与HIPT模型相关的工具函数脚本  
│   ├── utils.py    // 通用工具函数脚本  
│   ├── vision_transformer.py    // 视觉Transformer相关的脚本  
│   ├── README.md    // 说明文档  
│   ├── requirements.txt    // 项目依赖文件  
│   └── work\    // 各种分析和可视化脚本  
│       ├── Before train\    // 训练前相关脚本   
│       │   ├── Before train.py    // 训练前的分析脚本  
│       │   ├── Before train2.py    // 训练前的分析脚本2  
│       │   ├── Before train3.py    // 训练前的分析脚本3  
│       │   ├── Before train4.py    // 训练前的分析脚本4  
│       │   ├── Validation dataset.py    // 验证数据集分析脚本  
│       │   └── Validation dataset 2.py    // 验证数据集分析脚本2  
│       ├── IHC\  
│       │   ├── IHC-A-cell segmentation.py    // IHC-A 区域细胞分割脚本  
│       │   ├── IHC-A-expression.py    // IHC-A 区域基因表达脚本  
│       │   ├── IHC-B-cell segmentation.py    // IHC-B 区域细胞分割脚本  
│       │   └── IHC-B-expression.py    // IHC-B 区域基因表达脚本  
│       └── ...  

## 使用方法

### 1. 预处理和缩放图像
```bash

python preprocess_image.py \
    --valid_path /path/to/images/ \
    --train_path /path/to/images/ \
    --output_path /path/to/images/
```

```bash

python rescale.py \
    --prefix /path/to/images/
```

### 2. 特征提取


```bash
python extract_features.py \
    --prefix /path/to/images/ \
    --h5ad-file /path/to/cells.h5ad \
    --device cuda \
    --batch-size 100
```

```bash
python merge_features.py \
    --input-dir /path/to/embedd \
    --output-file /path/to/cell_features.pkl
```

### 3. 模型训练

```bash
python train_model.py \
    --feature_file /path/to/cell_features.pkl \
    --h5ad_file /path/to/gene_expression.h5ad \
    --gene_list /path/to/gene_list.txt \
    --output_model /path/to/mlp_model.pkl \
    --model_type mlp \
    --hidden_layers "512,512,1024,1024" \
    --alpha 0.01
```

### 4. 基因表达预测

```bash
python predict_expression.py \
    --model /path/to/mlp_model.pkl \
    --feature_file /path/to/new_cell_features.pkl \
    --output_file /path/to/predictions.h5ad \
```