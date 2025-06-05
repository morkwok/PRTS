import cv2
import numpy as np
from skimage import exposure

def match_histograms(source, template):
    """
    匹配两个图像的直方图，使源图像的颜色分布接近模板图像
    """
    result = np.zeros_like(source)
    for d in range(source.shape[-1]):
        result[..., d] = exposure.match_histograms(source[..., d], template[..., d])
    return result

def adjust_contrast(image, alpha=1.0, beta=0):
    """
    调整图像的对比度和亮度
    """
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def preprocess_images(valid_path, train_path, output_path):
    # 读取图像
    valid_img = cv2.imread(valid_path)
    train_img = cv2.imread(train_path)

    if valid_img is None or train_img is None:
        print("无法读取图像，请检查文件路径。")
        return

    # 匹配直方图
    matched_img = match_histograms(valid_img, train_img)

    # 计算对比度调整参数（这里简单使用平均亮度比）
    valid_mean = np.mean(valid_img)
    train_mean = np.mean(train_img)
    alpha = train_mean / valid_mean
    beta = 0

    # 调整对比度
    adjusted_img = adjust_contrast(matched_img, alpha=alpha, beta=beta)

    # 保存处理后的图像
    cv2.imwrite(output_path, adjusted_img)
    print(f"处理后的图像已保存到 {output_path}")

if __name__ == "__main__":
    valid_path = "valid\Fresh_Frozen.tif"
    train_path = "D:\IAIS\graduate\med\istar\istar-master\Visium_HD_Mouse_Brain_tissue_image.tif"
    output_path = "valid\FreshFrozen1.tif"
    preprocess_images(valid_path, train_path, output_path)