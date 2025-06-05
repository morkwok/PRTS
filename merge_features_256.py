import os
import pickle
import argparse
import glob
import numpy as np

def load_batch_file(file_path):
    """
    加载单个批次的pickle文件
    参数:
        file_path: pickle文件路径
    返回:
        批次数据字典
    """
    print(f"加载文件: {file_path}")
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def merge_cls_features(prefix):
    cls_folder = os.path.join(prefix, 'embedd', 'cls')
    if not os.path.exists(cls_folder):
        print(f"错误: 目录 {cls_folder} 不存在")
        return

    batch_files = sorted(glob.glob(os.path.join(cls_folder, "*_cls.pickle")))
    if not batch_files:
        print(f"未找到任何 cls 批次文件: {cls_folder}/*_cls.pickle")
        return

    print(f"找到 {len(batch_files)} 个 cls 批次文件")

    # 初始化合并数据结构
    merged_data = {
        'cell_ids': [],
        'features': []
    }

    # 处理所有批次
    total_cells = 0
    for batch_file in batch_files:
        batch_data = load_batch_file(batch_file)
        
        # 添加细胞ID
        merged_data['cell_ids'].extend(batch_data['cell_ids'])
        
        # 添加特征
        merged_data['features'].append(batch_data['features'])
        
        total_cells += len(batch_data['cell_ids'])
        print(f"已处理 {total_cells} 个细胞")

    # 转换为numpy数组
    merged_data['cell_ids'] = np.array(merged_data['cell_ids'])
    merged_data['features'] = np.concatenate(merged_data['features'], axis=0)
    print(f"cls 特征形状: {merged_data['features'].shape}")

    # 保存合并后的特征
    output_file = os.path.join(prefix, 'embedd', 'merged_cls_features.pickle')
    with open(output_file, 'wb') as f:
        pickle.dump(merged_data, f)

    print(f"合并完成！总共处理了 {total_cells} 个细胞，结果已保存到 {output_file}")

def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix', type=str, help='数据目录前缀')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    merge_cls_features(args.prefix)