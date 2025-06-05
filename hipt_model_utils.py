# 导入必要的库

# 线性代数/统计/绘图相关依赖
import numpy as np
from PIL import Image  # 图像处理

# PyTorch相关依赖
import torch
import torch.multiprocessing
from torchvision import transforms  # 图像变换
from einops import rearrange  # 张量操作

# 本地依赖
import vision_transformer as vits  # ViT-256模型
import vision_transformer4k as vits4k  # ViT-4K模型


# 设置多进程共享策略为文件系统
torch.multiprocessing.set_sharing_strategy("file_system")


def get_vit256(pretrained_weights=None, arch="vit_small", device=torch.device("cuda:0")):
    r"""
    构建ViT-256模型
    
    参数:
    - pretrained_weights (str): ViT-256模型预训练权重路径
    - arch (str): 模型架构名称，默认为"vit_small"
    - device (torch): 模型运行的设备，默认为cuda:0
    
    返回:
    - model256 (torch.nn): 初始化好的ViT-256模型
    """
    
    checkpoint_key = "teacher"  # 检查点字典中的键名
    # 自动检测可用设备
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    
    # 初始化ViT-256模型
    model256 = vits.__dict__[arch](patch_size=16, num_classes=0)
    
    # 冻结所有参数(不训练)
    for p in model256.parameters():
        p.requires_grad = False
    model256.eval()  # 设置为评估模式
    model256.to(device)  # 移动到指定设备

    # 如果提供了预训练权重
    if pretrained_weights is not None:
        # 加载状态字典(在CPU上)
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        
        # 处理检查点键
        if checkpoint_key is not None and checkpoint_key in state_dict:
            state_dict = state_dict[checkpoint_key]
            
        # 移除'module.'前缀(多GPU训练时添加的)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # 移除'backbone.'前缀(多裁剪包装器添加的)
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        
        # 加载状态字典(不严格要求完全匹配)
        model256.load_state_dict(state_dict, strict=False)
        
    return model256


def get_vit4k(pretrained_weights=None, arch="vit4k_xs", device=torch.device("cuda:0")):
    r"""
    构建ViT-4K模型
    
    参数:
    - pretrained_weights (str): ViT-4K模型预训练权重路径
    - arch (str): 模型架构名称，默认为"vit4k_xs"
    - device (torch): 模型运行的设备，默认为cuda:0
    
    返回:
    - model4k (torch.nn): 初始化好的ViT-4K模型
    """
    
    checkpoint_key = "teacher"  # 检查点字典中的键名
    # 自动检测可用设备
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    
    # 初始化ViT-4K模型
    model4k = vits4k.__dict__[arch](num_classes=0)
    
    # 冻结所有参数(不训练)
    for p in model4k.parameters():
        p.requires_grad = False
    model4k.eval()  # 设置为评估模式
    model4k.to(device)  # 移动到指定设备

    # 如果提供了预训练权重
    if pretrained_weights is not None:
        # 加载状态字典(在CPU上)
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        
        # 处理检查点键
        if checkpoint_key is not None and checkpoint_key in state_dict:
            state_dict = state_dict[checkpoint_key]
            
        # 移除'module.'前缀(多GPU训练时添加的)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # 移除'backbone.'前缀(多裁剪包装器添加的)
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        
        # 加载状态字典(不严格要求完全匹配)
        model4k.load_state_dict(state_dict, strict=False)
        
    return model4k


def eval_transforms():
    """
    创建用于评估的图像变换管道
    
    返回:
    - eval_t (torchvision.transforms.Compose): 包含标准化处理的图像变换
    """
    mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)  # 归一化参数
    eval_t = transforms.Compose([
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=mean, std=std)  # 标准化
    ])
    return eval_t


def roll_batch2img(batch: torch.Tensor, w: int, h: int, patch_size=256):
    """
    将图像张量批次(一批[256x256]图像)转换为[W x H]的PIL.Image对象
    
    参数:
    - batch (torch.Tensor): [B x 3 x 256 x 256]图像张量批次
    - w (int): 输出图像的宽度(以patch为单位)
    - h (int): 输出图像的高度(以patch为单位)
    - patch_size (int): 每个patch的大小，默认为256
    
    返回:
    - Image.PIL: [W x H X 3]图像
    """
    # 重塑张量形状
    batch = batch.reshape(w, h, 3, patch_size, patch_size)
    # 重新排列维度并合并patch
    img = rearrange(batch, "p1 p2 c w h-> c (p1 w) (p2 h)").unsqueeze(dim=0)
    # 转换为PIL图像
    return Image.fromarray(tensorbatch2im(img)[0])


def tensorbatch2im(input_image, imtype=np.uint8):
    r"""
    将张量数组转换为numpy图像数组
    
    参数:
    - input_image (torch.Tensor): (B, C, W, H)张量
    - imtype (type): 转换后的numpy数组类型，默认为np.uint8
    
    返回:
    - image_numpy (np.array): (B, W, H, C) numpy数组
    """
    if not isinstance(input_image, np.ndarray):
        # 转换为numpy数组
        image_numpy = input_image.cpu().float().numpy()
        # 转置维度并缩放值到0-255范围
        image_numpy = (np.transpose(image_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
    else:
        # 如果已经是numpy数组，直接返回
        image_numpy = input_image
    return image_numpy.astype(imtype)