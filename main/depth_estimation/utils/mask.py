import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os

def robust_reflection_detection(image):
    """
    结合多种物理特征的方法
    在实际深度估计任务中表现良好
    """
    # 1. 亮度通道分析
    if image.shape[1] == 3:
        gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
    else:
        gray = image
    
    # 2. 计算多个特征
    # 特征1: 亮度百分位（避免固定阈值）
    bright_thresh = torch.quantile(gray, 0.95)  # 最亮的5%
    very_bright = gray > bright_thresh
    
    # 特征2: 局部对比度（反射区域纹理少）
    local_std = F.avg_pool2d(gray**2, 3, stride=1, padding=1) - \
                F.avg_pool2d(gray, 3, stride=1, padding=1)**2
    low_texture = local_std < 0.02
    
    # 特征3: 颜色一致性（反射区域颜色分布异常）
    if image.shape[1] == 3:
        r, g, b = image[:, 0:1], image[:, 1:2], image[:, 2:3]
        color_std = torch.std(torch.cat([r, g, b], dim=1), dim=1, keepdim=True)
        color_anomaly = color_std < 0.1  # 颜色变化小可能是反射
        
    # 特征4: 边缘密度（反射区域内部边缘少）
    sobel_x = F.conv2d(gray, torch.tensor([[[[1,0,-1],[2,0,-2],[1,0,-1]]]], 
                                         device=gray.device)/8.0, padding=1)
    sobel_y = F.conv2d(gray, torch.tensor([[[[1,2,1],[0,0,0],[-1,-2,-1]]]], 
                                         device=gray.device)/8.0, padding=1)
    edge_mag = torch.sqrt(sobel_x**2 + sobel_y**2 + 1e-6)
    low_edges = edge_mag < 0.05
    
    # 3. 组合特征（加权投票）
    reflection_score = (
        very_bright.float() * 0.4 +
        low_texture.float() * 0.3 +
        low_edges.float() * 0.2
    )
    
    if image.shape[1] == 3:
        reflection_score += color_anomaly.float() * 0.1
    
    # 4. 后处理：形态学操作
    reflection_mask = (reflection_score > 0.6).float()
    reflection_mask = F.max_pool2d(reflection_mask, 3, stride=1, padding=1)
    reflection_mask = F.avg_pool2d(reflection_mask, 5, stride=1, padding=2) > 0.3
    
    return reflection_mask.float()

def save_results(img_np, mask_np, output_dir="output"):
    """保存结果到文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存原始图片
    img_pil = Image.fromarray(img_np.astype(np.uint8))
    img_pil.save(os.path.join(output_dir, "original.png"))
    
    # 保存掩码（归一化到0-255）
    mask_normalized = (mask_np * 255).astype(np.uint8)
    mask_pil = Image.fromarray(mask_normalized)
    mask_pil.save(os.path.join(output_dir, "reflection_mask.png"))
    
    # 保存叠加效果
    overlay = img_np.copy()
    red_mask = mask_np > 0.5
    overlay[..., 0] = np.where(red_mask, 255, overlay[..., 0])
    overlay[..., 1] = np.where(red_mask, 0, overlay[..., 1])
    overlay[..., 2] = np.where(red_mask, 0, overlay[..., 2])
    
    overlay_pil = Image.fromarray(overlay.astype(np.uint8))
    overlay_pil.save(os.path.join(output_dir, "overlay.png"))
    
    print(f"结果已保存到目录: {output_dir}/")
    print(f"  - original.png (原始图片)")
    print(f"  - reflection_mask.png (反光掩码)")
    print(f"  - overlay.png (叠加效果)")

def main():
    """主函数：使用你自己的图片测试"""
    # 1. 读取图片
    image_path = "data/example_dataset/J/01189.png"  # 你的图片路径
    
    try:
        # 打开图片
        img = Image.open(image_path).convert('RGB')
        
        # 调整大小（可选）
        if img.size[0] > 800 or img.size[1] > 600:
            img = img.resize((512, 384))
        
        print(f"图片尺寸: {img.size}")
        
        # 转换为numpy
        img_np = np.array(img)
        
        # 转换为tensor进行处理
        img_tensor = torch.from_numpy(img_np).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        
        print(f"Tensor形状: {img_tensor.shape}")
        
        # 检测反光区域
        with torch.no_grad():
            reflection_mask = robust_reflection_detection(img_tensor)
        
        # 转换为numpy
        mask_np = reflection_mask.squeeze().cpu().numpy()
        
        # 打印统计信息
        reflection_pixels = mask_np.sum()
        total_pixels = mask_np.size
        reflection_ratio = reflection_pixels / total_pixels * 100
        
        print(f"\n反光检测结果:")
        print(f"  反光像素数: {int(reflection_pixels)} / {total_pixels}")
        print(f"  反光比例: {reflection_ratio:.2f}%")
        print(f"  平均反射分数: {mask_np.mean():.4f}")
        print(f"  最大反射分数: {mask_np.max():.4f}")
        
        # 保存结果
        save_results(img_np, mask_np)
        
        # 额外：打印更详细的统计
        print(f"\n详细统计:")
        thresholds = [0.3, 0.5, 0.7, 0.9]
        for thresh in thresholds:
            area = (mask_np > thresh).sum()
            ratio = area / total_pixels * 100
            print(f"  阈值>{thresh}: {int(area)}像素 ({ratio:.2f}%)")
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{image_path}'")
    except Exception as e:
        print(f"处理图片时出错: {e}")

if __name__ == "__main__":
    main()