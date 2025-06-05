import cv2
import numpy as np
from skimage import exposure
import argparse

def match_histograms(source, template):

    result = np.zeros_like(source)
    for d in range(source.shape[-1]):
        result[..., d] = exposure.match_histograms(source[..., d], template[..., d])
    return result

def adjust_contrast(image, alpha=1.0, beta=0):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def preprocess_images(valid_path, train_path, output_path):
    valid_img = cv2.imread(valid_path)
    train_img = cv2.imread(train_path)

    if valid_img is None or train_img is None:
        print("无法读取图像，请检查文件路径。")
        return

    matched_img = match_histograms(valid_img, train_img)

    valid_mean = np.mean(valid_img)
    train_mean = np.mean(train_img)
    alpha = train_mean / valid_mean
    beta = 0

    adjusted_img = adjust_contrast(matched_img, alpha=alpha, beta=beta)

    cv2.imwrite(output_path, adjusted_img)
    print(f"处理后的图像已保存到 {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocess')
    parser.add_argument('valid_path', type=str, help='验证图像的路径')
    parser.add_argument('train_path', type=str, help='训练图像的路径')
    parser.add_argument('output_path', type=str, help='输出图像的路径')
    args = parser.parse_args()
    preprocess_images(args.valid_path, args.train_path, args.output_path)