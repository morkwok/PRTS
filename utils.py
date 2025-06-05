import itertools
from PIL import Image
import pickle
import os

import numpy as np 
import pandas as pd 
import yaml


Image.MAX_IMAGE_PIXELS = None


def mkdir(path):

    dirname = os.path.dirname(path)
    if dirname != '':
        os.makedirs(dirname, exist_ok=True)


def load_image(filename, verbose=True):

    img = Image.open(filename)
    img = np.array(img)
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]  # 移除alpha通道(透明度通道)
    if verbose:
        print(f'Image loaded from {filename}')
    return img


def load_mask(filename, verbose=True):
    """
    加载并处理掩码图像
    参数:
        filename: 掩码文件路径
        verbose: 是否打印加载信息
    返回:
        二值化的掩码数组(True表示有效区域)
    """
    mask = load_image(filename, verbose=verbose)
    mask = mask > 0  # 二值化
    if mask.ndim == 3:  # 如果是RGB图像
        mask = mask.any(2)  # 任意通道为True则为True
    return mask


def save_image(img, filename):
    """
    保存图像文件
    参数:
        img: numpy数组格式的图像数据
        filename: 保存路径
    """
    mkdir(filename)
    Image.fromarray(img).save(filename)
    print(filename)


def read_lines(filename):
    """
    读取文本文件所有行
    参数:
        filename: 文件路径
    返回:
        去除行尾空格的字符串列表
    """
    with open(filename, 'r') as file:
        lines = [line.rstrip() for line in file]
    return lines


def read_string(filename):
    """
    读取文本文件第一行内容
    参数:
        filename: 文件路径
    返回:
        文件第一行内容
    """
    return read_lines(filename)[0]


def write_lines(strings, filename):
    """
    写入多行文本到文件
    参数:
        strings: 字符串列表
        filename: 文件路径
    """
    mkdir(filename)
    with open(filename, 'w') as file:
        for s in strings:
            file.write(f'{s}\n')
    print(filename)


def write_string(string, filename):
    """
    写入单行文本到文件
    参数:
        string: 要写入的字符串
        filename: 文件路径
    """
    return write_lines([string], filename)


def save_pickle(x, filename):
    """
    使用pickle序列化保存Python对象
    参数:
        x: 要保存的Python对象
        filename: 保存路径
    """
    mkdir(filename)
    with open(filename, 'wb') as file:
        pickle.dump(x, file)
    print(filename)


def load_pickle(filename, verbose=True):
    """
    加载pickle序列化的Python对象
    参数:
        filename: 文件路径
        verbose: 是否打印加载信息
    返回:
        反序列化的Python对象
    """
    with open(filename, 'rb') as file:
        x = pickle.load(file)
    if verbose:
        print(f'Pickle loaded from {filename}')
    return x


def load_tsv(filename, index=True):
    """
    加载TSV(制表符分隔)文件为DataFrame
    参数:
        filename: 文件路径
        index: 是否使用第一列作为索引
    返回:
        pandas DataFrame对象
    """
    if index:
        index_col = 0
    else:
        index_col = None
    df = pd.read_csv(filename, sep='\t', header=0, index_col=index_col)
    print(f'Dataframe loaded from {filename}')
    return df


def save_tsv(x, filename, **kwargs):
    """
    保存DataFrame为TSV文件
    参数:
        x: pandas DataFrame对象
        filename: 保存路径
        **kwargs: 额外参数传递给to_csv
    """
    mkdir(filename)
    if 'sep' not in kwargs.keys():
        kwargs['sep'] = '\t'  # 默认使用制表符分隔
    x.to_csv(filename, **kwargs)
    print(filename)


def load_yaml(filename, verbose=False):
    """
    加载YAML配置文件
    参数:
        filename: 文件路径
        verbose: 是否打印加载信息
    返回:
        解析后的Python对象
    """
    with open(filename, 'r') as file:
        content = yaml.safe_load(file)
    if verbose:
        print(f'YAML loaded from {filename}')
    return content


def save_yaml(filename, content):
    """
    保存Python对象为YAML文件
    参数:
        filename: 文件路径
        content: 要保存的Python对象
    """
    with open(filename, 'w') as file:
        yaml.dump(content, file)
    print(file)


def join(x):
    """
    将嵌套列表展平为一维列表
    参数:
        x: 嵌套列表
    返回:
        展平后的一维列表
    """
    return list(itertools.chain.from_iterable(x))


def get_most_frequent(x):
    """
    获取数组中出现最频繁的元素
    参数:
        x: 输入数组
    返回:
        出现次数最多的元素
    """
    # 获取唯一值和计数
    uniqs, counts = np.unique(x, return_counts=True)
    return uniqs[counts.argmax()]  # 返回计数最大的元素


def sort_labels(labels, descending=True):
    """
    对标签进行排序和重新编号
    参数:
        labels: 原始标签数组
        descending: 是否按降序排列(默认True)
    返回:
        (重新编号后的标签数组, 排序后的唯一标签)
    """
    labels = labels.copy()
    isin = labels >= 0  # 有效标签掩码
    # 获取唯一标签、重新编号和计数
    labels_uniq, labels[isin], counts = np.unique(
            labels[isin], return_inverse=True, return_counts=True)
    c = counts
    if descending:
        c = c * (-1)  # 降序排列
    order = c.argsort()  # 获取排序顺序
    rank = order.argsort()  # 获取排名
    labels[isin] = rank[labels[isin]]  # 应用新的编号
    return labels, labels_uniq[order]  # 返回新标签和排序后的唯一标签