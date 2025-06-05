import os
from time import time
import argparse
import pickle

from einops import rearrange, reduce, repeat
import numpy as np
import skimage
import torch
import pandas as pd
import anndata as ad

from utils import load_image
from utils import load_pickle, save_pickle, join
from PIL import Image
from einops import rearrange

import torch.multiprocessing
from torchvision import transforms

from model_utils import (
    get_vit256,
    tensorbatch2im,
    eval_transforms
)

def load_cell_ids(prefix, h5ad_file=None):

    if h5ad_file:
        print(f"从h5ad文件读取细胞ID: {h5ad_file}")
        adata = ad.read_h5ad(h5ad_file)
        cell_ids = adata.obs.index.tolist()
    else:
        print(f"从图像目录中读取所有细胞ID: {prefix}cells_256/")
        cells_dir = f'{prefix}cells_256/'
        if not os.path.exists(cells_dir):
            raise FileNotFoundError(f"目录不存在: {cells_dir}")
        
        cell_ids = []
        files = os.listdir(cells_dir)
        for file in files:
            if file.endswith('_256.tif'):
                cell_id = file.replace('_256.tif', '')
                cell_ids.append(cell_id)
        
        print(f"在{cells_dir}中找到{len(cell_ids)}个细胞图像")
    
    return cell_ids

def get_data_batch(prefix, cell_ids, batch_idx, batch_size):
    start_idx = batch_idx * batch_size
    end_idx = min(start_idx + batch_size, len(cell_ids))
    batch_cell_ids = cell_ids[start_idx:end_idx]

    embs_256 = [] 
    
    print(f"加载第{batch_idx+1}批细胞图像 ({start_idx}:{end_idx})...")
    for cell_id in batch_cell_ids:
        global_path = f'{prefix}/{cell_id}_256.tif'
        
        if not os.path.exists(global_path):
            print(f"警告: 细胞 {cell_id} 的图像文件不存在，跳过")
            continue

        img_256 = load_image(global_path)
        img_256 = img_256.astype(np.float32) / 255.0
        embs_256.append(img_256)

    if len(embs_256) == 0:
        raise ValueError(f"批次 {batch_idx} 中没有找到有效的细胞图像")
        
    embs_256 = np.stack(embs_256)  
    
    return embs_256, batch_cell_ids

def extract_and_process_features(model, embs_256, args):
    model = model.to(args.device)
    model.eval()
    
    features = {}
    with torch.no_grad():
        imgs = torch.stack([eval_transforms()(img) for img in embs_256]).to(args.device)
        fea_all256 = model.forward_all(imgs).cpu()
        cls_features = fea_all256[:, 0].numpy()  
        sub_feat = fea_all256[:, 1:]
        start_h = (16 - 2) // 2
        start_w = (16 - 2) // 2
        middle_patches = sub_feat[:, start_h:start_h+2, start_w:start_w+2, :]
        middle_patches = rearrange(middle_patches, 'b h w c -> b c h w')
        sub_features = reduce(middle_patches, 'b c h w -> b c', 'mean')
        sub_features = rearrange(sub_features, 'b (c1 c2) -> b c1 c2', c2=2)
        sub_features = reduce(sub_features, 'b c1 c2 -> b c1', 'mean')
        sub_features = sub_features.numpy()
        fused_features = np.concatenate((cls_features, sub_features), axis=1)
        features['cls'] = cls_features
        features['sub'] = sub_features
        features['fused'] = fused_features

    print("特征提取完成。")
    print(f"cls特征形状: {features['cls'].shape} - (细胞数量, 384)")
    print(f"sub特征形状: {features['sub'].shape} - (细胞数量, 192)")
    print(f"融合特征形状: {features['fused'].shape} - (细胞数量, 576)")

    return features

def save_batch_features(features, cell_ids, output_prefix, batch_idx):

    cls_features = features['cls']
    sub_features = features['sub']

    cls_folder = os.path.join(output_prefix, 'embedd', 'cls')
    sub_folder = os.path.join(output_prefix, 'embedd', 'sub')
    os.makedirs(cls_folder, exist_ok=True)
    os.makedirs(sub_folder, exist_ok=True)

    cls_data = {
        'cell_ids': cell_ids,
        'features': cls_features
    }
    cls_output_file = os.path.join(cls_folder, f"batch_{batch_idx:04d}_cls.pickle")
    with open(cls_output_file, 'wb') as f:
        pickle.dump(cls_data, f)
    print(f"批次 {batch_idx} 的 cls 特征已保存到: {cls_output_file}")

    sub_data = {
        'cell_ids': cell_ids,
        'features': sub_features
    }
    sub_output_file = os.path.join(sub_folder, f"batch_{batch_idx:04d}_sub.pickle")
    with open(sub_output_file, 'wb') as f:
        pickle.dump(sub_data, f)
    print(f"批次 {batch_idx} 的 sub 特征已保存到: {sub_output_file}")

    fused_data = {
        'cell_ids': cell_ids,
        'features': fused_features
    }
    fused_folder = os.path.join(output_prefix, 'embedd', 'fused')
    os.makedirs(fused_folder, exist_ok=True)
    fused_output_file = os.path.join(fused_folder, f"batch_{batch_idx:04d}_fused.pickle")
    with open(fused_output_file, 'wb') as f:
        pickle.dump(fused_data, f)
    print(f"批次 {batch_idx} 的融合特征已保存到: {fused_output_file}")

def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix', type=str, help='数据目录前缀')
    parser.add_argument('--h5ad-file', type=str, default=None,
                      help='包含细胞ID的h5ad文件路径（可选）')
    parser.add_argument('--batch-size', type=int, default=256,
                      help='批处理大小')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--random-weights', action='store_true',
                      help='是否使用随机初始化的权重')
    parser.add_argument('--model256-path', type=str, 
                      default='checkpoints/vit256_small_dino.pth',
                      help='ViT-256模型权重路径')
    return parser.parse_args()

def main():
    """主处理流程"""
    args = get_args()
    np.random.seed(0)
    torch.manual_seed(0)

    cell_ids = load_cell_ids(args.prefix, args.h5ad_file)

    n_cells = len(cell_ids)
    n_batches = (n_cells + args.batch_size - 1) // args.batch_size
    print(f"总细胞数: {n_cells}")
    print(f"批次大小: {args.batch_size}")
    print(f"总批次数: {n_batches}")

    device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    if args.random_weights:
        print("使用随机初始化权重")
        model256_path = None
    else:
        print(f"加载预训练权重:")
        print(f"ViT-256模型: {args.model256_path}")
        model256_path = args.model256_path
        
        if not os.path.exists(model256_path):
            raise FileNotFoundError(f"找不到ViT-256模型权重文件: {model256_path}")

    output_prefix = args.prefix
    print(f"\n开始处理数据...")

    for batch_idx in range(n_batches):
        print(f"\n处理批次 {batch_idx+1}/{n_batches}")

        embs_256, batch_cell_ids = get_data_batch(
            args.prefix, cell_ids, batch_idx, args.batch_size)

        features = extract_and_process_features(model256, embs_256, args)

        save_batch_features(features, batch_cell_ids, output_prefix, batch_idx)

        del embs_256, features
        torch.cuda.empty_cache()
        
        print(f"批次 {batch_idx+1} 处理完成")

    print(f"\n所有数据处理完成")
    print(f"特征已保存到: {os.path.join(output_prefix, 'embedd')} 目录中")

if __name__ == '__main__':
    main()