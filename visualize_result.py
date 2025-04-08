import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

def visualize_keypoints(img_path, kpts_path, output_path, num_points=100):
    # 读取原始图像
    img = Image.open(img_path)
    img = np.array(img)
    
    # 读取关键点数据
    data = np.load(kpts_path)
    keypoints = data['keypoints']  # N x 3 (x, y, scale)
    scores = data['scores']  # N
    
    # 随机选择10个点
    if len(keypoints) > num_points:
        indices = random.sample(range(len(keypoints)), num_points)
        keypoints = keypoints[indices]
        scores = scores[indices]
    
    # 创建图像
    plt.figure(figsize=(12,8))
    plt.imshow(img, cmap='gray')
    
    # 绘制关键点
    for kp, score in zip(keypoints, scores):
        x, y, scale = kp
        plt.scatter(x, y, s=scale*50*score, c='r', alpha=0.6)
    
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("Visualize R2D2 keypoints")
    parser.add_argument("--img", type=str, default="imgs/brooklyn.png")
    parser.add_argument("--kpts", type=str, default=None)
    args = parser.parse_args()
    
    if args.kpts is None:
        args.kpts = args.img + '.r2d2'
        
    visualize_keypoints(args.img, args.kpts, args.kpts + '.viz.png')
