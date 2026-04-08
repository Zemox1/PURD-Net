import os
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple
import time

# ===================== 配置 =====================
GT_DIR = "data/example_dataset/depth"  # 真实深度图(GT)目录
YOUR_MODEL_DIR = "infer_depth/depth_maps"  # 你的模型深度图
OTHER_MODEL_DIR = "depthanything"  # 其他模型深度图
START_NUM = 1100
END_NUM = 1315

# ===================== 工具函数 =====================
def load_depth_image(path: str) -> np.ndarray:
    """加载并归一化深度图"""
    if not os.path.exists(path):
        return None
    img = Image.open(path).convert('L')
    depth_np = np.array(img, dtype=np.float32)
    if depth_np.max() > 1.0:
        depth_np = depth_np / 255.0
    return np.nan_to_num(depth_np, nan=0.0, posinf=1.0, neginf=0.0)

def calculate_all_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """
    计算所有7个深度估计指标
    返回: 包含7个指标的字典
    """
    # 有效掩码（排除背景0值）
    valid_mask = (gt > 1e-6) & (pred > 1e-6)
    if not np.any(valid_mask):
        return {k: np.nan for k in ["rmse", "mae", "absrel", "sqrel", "delta1", "delta2", "delta3"]}
    
    pred_valid = pred[valid_mask]
    gt_valid = gt[valid_mask]
    
    # 1. RMSE (均方根误差)
    mse = np.mean((pred_valid - gt_valid) ** 2)
    rmse = np.sqrt(mse)
    
    # 2. MAE (平均绝对误差)
    mae = np.mean(np.abs(pred_valid - gt_valid))
    
    # 3. AbsRel (绝对相对误差)
    absrel = np.mean(np.abs(pred_valid - gt_valid) / (gt_valid + 1e-6))
    
    # 4. SqRel (平方相对误差)
    sqrel = np.mean(((pred_valid - gt_valid) ** 2) / (gt_valid + 1e-6))
    
    # 5-7. delta1, delta2, delta3 (阈值精度)
    ratio = np.maximum(pred_valid / (gt_valid + 1e-6), gt_valid / (pred_valid + 1e-6))
    delta1 = np.mean(ratio < 1.25) * 100  # δ < 1.25
    delta2 = np.mean(ratio < 1.25 ** 2) * 100  # δ < 1.25^2
    delta3 = np.mean(ratio < 1.25 ** 3) * 100  # δ < 1.25^3
    
    return {
        "rmse": rmse,
        "mae": mae,
        "absrel": absrel,
        "sqrel": sqrel,
        "delta1": delta1,
        "delta2": delta2,
        "delta3": delta3
    }

def batch_calculate_metrics() -> Dict[str, Dict]:
    """批量计算所有指标"""
    metrics = {
        "your_model": {"rmse": [], "mae": [], "absrel": [], "sqrel": [], 
                       "delta1": [], "delta2": [], "delta3": []},
        "other_model": {"rmse": [], "mae": [], "absrel": [], "sqrel": [], 
                       "delta1": [], "delta2": [], "delta3": []},
        "valid_nums": []
    }
    
    valid_count = 0
    total_count = END_NUM - START_NUM + 1
    
    for num in range(START_NUM, END_NUM + 1):
        img_name = f"{num:05d}.png"
        gt_path = os.path.join(GT_DIR, img_name)
        your_path = os.path.join(YOUR_MODEL_DIR, img_name)
        other_path = os.path.join(OTHER_MODEL_DIR, img_name)
        
        # 检查文件是否存在
        if not all(os.path.exists(p) for p in [gt_path, your_path, other_path]):
            print(f"跳过 {img_name}: 文件缺失")
            continue
        
        # 加载图像
        gt_depth = load_depth_image(gt_path)
        your_depth = load_depth_image(your_path)
        other_depth = load_depth_image(other_path)
        
        if any(d is None for d in [gt_depth, your_depth, other_depth]):
            continue
            
        # 计算指标
        your_metrics = calculate_all_metrics(your_depth, gt_depth)
        other_metrics = calculate_all_metrics(other_depth, gt_depth)
        
        # 检查是否有NaN值
        if any(np.isnan(v) for v in your_metrics.values()) or \
           any(np.isnan(v) for v in other_metrics.values()):
            continue
        
        # 保存结果
        for key in your_metrics.keys():
            metrics["your_model"][key].append(your_metrics[key])
            metrics["other_model"][key].append(other_metrics[key])
        
        metrics["valid_nums"].append(num)
        valid_count += 1
        
        # 打印进度
        if valid_count % 20 == 0:
            print(f"已处理 {valid_count}/{total_count} 张图片...")
    
    return metrics, valid_count

def print_metrics_summary(metrics: Dict[str, Dict], valid_count: int):
    """打印指标汇总"""
    print("\n" + "="*100)
    print(f"📊 深度估计评估结果 (有效图片: {valid_count}张)")
    print("="*100)
    
    # 表头
    print(f"{'指标':<10} {'单位':<8} {'方向':<6} {'你的模型':<12} {'其他模型':<12} {'相对变化':<12} {'胜出模型':<8}")
    print("-"*100)
    
    # 定义每个指标的信息
    metric_info = {
        "rmse":   ("RMSE",    "",     "↓",   "rmse越低越好"),
        "mae":    ("MAE",     "",     "↓",   "mae越低越好"),
        "absrel": ("AbsRel",  "",     "↓",   "absrel越低越好"),
        "sqrel":  ("SqRel",   "",     "↓",   "sqrel越低越好"),
        "delta1": ("δ1",      "%",    "↑",   "delta1越高越好"),
        "delta2": ("δ2",      "%",    "↑",   "delta2越高越好"),
        "delta3": ("δ3",      "%",    "↑",   "delta3越高越好"),
    }
    
    results_comparison = []
    
    for key, (name, unit, direction, desc) in metric_info.items():
        your_mean = np.mean(metrics["your_model"][key])
        other_mean = np.mean(metrics["other_model"][key])
        
        # 计算相对变化
        if direction == "↓":  # 越小越好
            change = (other_mean - your_mean) / other_mean * 100
            better_model = "你的模型" if your_mean < other_mean else "其他模型"
        else:  # 越大越好 (delta指标)
            change = (your_mean - other_mean) / other_mean * 100
            better_model = "你的模型" if your_mean > other_mean else "其他模型"
        
        # 格式化输出
        your_str = f"{your_mean:.6f}{unit}"
        other_str = f"{other_mean:.6f}{unit}"
        change_str = f"{change:+.2f}%"
        
        # 颜色标记
        if (direction == "↓" and your_mean < other_mean) or (direction == "↑" and your_mean > other_mean):
            change_str = f"✅ {change_str}"
        else:
            change_str = f"❌ {change_str}"
        
        print(f"{name:<10} {unit:<8} {direction:<6} {your_str:<12} {other_str:<12} {change_str:<12} {better_model:<8}")
        
        # 保存比较结果
        results_comparison.append({
            "metric": name,
            "your": your_mean,
            "other": other_mean,
            "change": change,
            "better": better_model
        })
    
    print("-"*100)
    
    # 统计胜出次数
    your_wins = sum(1 for r in results_comparison if r["better"] == "你的模型")
    other_wins = len(results_comparison) - your_wins
    
    print(f"\n🎯 综合对比: 你的模型在 {your_wins}/7 个指标上领先，其他模型在 {other_wins}/7 个指标上领先")
    
    # 关键指标分析
    print("\n🔍 关键指标分析:")
    print(f"  1. AbsRel: 你的模型 {np.mean(metrics['your_model']['absrel']):.6f} vs "
          f"其他模型 {np.mean(metrics['other_model']['absrel']):.6f}")
    print(f"  2. δ1: 你的模型 {np.mean(metrics['your_model']['delta1']):.2f}% vs "
          f"其他模型 {np.mean(metrics['other_model']['delta1']):.2f}%")
    print(f"  3. RMSE: 你的模型 {np.mean(metrics['your_model']['rmse']):.6f} vs "
          f"其他模型 {np.mean(metrics['other_model']['rmse']):.6f}")

def print_detailed_statistics(metrics: Dict[str, Dict]):
    """打印详细统计信息"""
    print("\n" + "="*100)
    print("📈 详细统计信息")
    print("="*100)
    
    for model_name in ["your_model", "other_model"]:
        print(f"\n{model_name.upper().replace('_', ' ')}:")
        print("-"*60)
        
        model_data = metrics[model_name]
        for key in model_data.keys():
            values = model_data[key]
            if not values:
                continue
                
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            
            print(f"{key.upper():<8}: 均值={mean_val:.6f}, 标准差={std_val:.6f}, "
                  f"范围=[{min_val:.6f}, {max_val:.6f}]")

# ===================== 主程序 =====================
if __name__ == "__main__":
    print("开始计算深度估计指标...")
    start_time = time.time()
    
    metrics, valid_count = batch_calculate_metrics()
    
    if valid_count == 0:
        print("❌ 无有效数据! 请检查文件路径和格式")
        exit()
    
    elapsed_time = time.time() - start_time
    print(f"\n✅ 计算完成! 耗时: {elapsed_time:.2f}秒")
    
    # 打印汇总结果
    print_metrics_summary(metrics, valid_count)
    
    # 打印详细统计
    print_detailed_statistics(metrics)
    
    print("\n" + "="*100)
    print("🎉 评估完成!")
    print("="*100)