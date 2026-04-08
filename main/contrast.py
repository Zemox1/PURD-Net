import os
import numpy as np
from PIL import Image
import torch
from typing import Dict, List

# ===================== 核心配置 =====================
# 路径配置（根据实际情况修改）
GT_DIR = "/data/wrj/newA/0.1-2I+t/data/example_dataset/depth"  # 真实深度图(GT)目录
YOUR_MODEL_DIR = "/data/wrj/newA/0.1-2I+t/infer_depth/depth_maps"  # 你的模型深度图
OTHER_MODEL_DIR = "/data/wrj/newA/0.1-2I+t/depthanything"  # 其他模型深度图

# 编号范围
START_NUM = 1100
END_NUM = 1315

# 设备（可选，若需要GPU加速）
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# ===================== 工具函数 =====================
def load_depth_image(path: str) -> np.ndarray:
    """
    加载深度图并转换为浮点型数组（处理不同格式的深度图）
    返回：归一化后的深度值数组（H, W）
    """
    if not os.path.exists(path):
        return None

    # 读取图片（支持PNG/JPG，灰度图）
    img = Image.open(path).convert('L')  # 转为灰度图
    depth_np = np.array(img, dtype=np.float32)

    # 归一化（统一到0-1范围，消除不同模型输出范围差异）
    if depth_np.max() > 1.0:  # 若像素值是0-255范围
        depth_np = depth_np / 255.0

    # 处理异常值（NaN/Inf）
    depth_np = np.nan_to_num(depth_np, nan=0.0, posinf=1.0, neginf=0.0)
    return depth_np


def calculate_rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    计算RMSE（均方根误差）
    仅计算有效区域（GT>0的像素，避免背景0值干扰）
    """
    # 只计算GT有效区域（排除背景0值）
    valid_mask = (gt > 0) & (pred > 0)
    if not np.any(valid_mask):
        return np.nan  # 无有效像素，返回NaN

    # 计算误差
    pred_valid = pred[valid_mask]
    gt_valid = gt[valid_mask]
    mse = np.mean((pred_valid - gt_valid) ** 2)
    rmse = np.sqrt(mse)
    return rmse


def batch_calculate_rmse() -> Dict[str, List[float]]:
    """
    批量计算1100-1315编号的RMSE
    返回：各模型的RMSE列表
    """
    # 存储每个编号的RMSE
    results = {
        "your_model": [],  # 你的模型RMSE
        "other_model": [],  # 其他模型RMSE
        "valid_nums": []  # 有效编号（所有文件都存在的编号）
    }

    # 遍历编号（1100→01100，补零5位）
    for num in range(START_NUM, END_NUM + 1):
        img_name = f"{num:05d}.png"

        # 构建各文件路径
        gt_path = os.path.join(GT_DIR, img_name)
        your_path = os.path.join(YOUR_MODEL_DIR, img_name)
        other_path = os.path.join(OTHER_MODEL_DIR, img_name)

        # 检查文件是否存在
        missing_files = []
        if not os.path.exists(gt_path):
            missing_files.append(f"GT: {gt_path}")
        if not os.path.exists(your_path):
            missing_files.append(f"你的模型: {your_path}")
        if not os.path.exists(other_path):
            missing_files.append(f"其他模型: {other_path}")

        if missing_files:
            print(f"跳过 {img_name}：缺失文件 {missing_files}")
            continue

        # 加载深度图
        gt_depth = load_depth_image(gt_path)
        your_depth = load_depth_image(your_path)
        other_depth = load_depth_image(other_path)

        # 检查加载结果
        if gt_depth is None or your_depth is None or other_depth is None:
            print(f"跳过 {img_name}：图片加载失败")
            continue

        # 确保尺寸一致（避免尺寸不匹配报错）
        if your_depth.shape != gt_depth.shape:
            # 调整尺寸到GT大小（双线性插值）
            from scipy.ndimage import zoom
            scale = (gt_depth.shape[0] / your_depth.shape[0], gt_depth.shape[1] / your_depth.shape[1])
            your_depth = zoom(your_depth, scale, order=1)  # 双线性插值
        if other_depth.shape != gt_depth.shape:
            scale = (gt_depth.shape[0] / other_depth.shape[0], gt_depth.shape[1] / other_depth.shape[1])
            other_depth = zoom(other_depth, scale, order=1)

        # 计算RMSE
        your_rmse = calculate_rmse(your_depth, gt_depth)
        other_rmse = calculate_rmse(other_depth, gt_depth)

        # 过滤NaN值
        if np.isnan(your_rmse) or np.isnan(other_rmse):
            print(f"跳过 {img_name}：无有效像素计算RMSE")
            continue

        # 保存结果
        results["your_model"].append(your_rmse)
        results["other_model"].append(other_rmse)
        results["valid_nums"].append(num)

        # 打印单张结果（可选）
        print(f"编号 {img_name} → 你的模型RMSE: {your_rmse:.6f} | 其他模型RMSE: {other_rmse:.6f}")

    return results


# ===================== 主流程 =====================
if __name__ == "__main__":
    # 批量计算RMSE
    results = batch_calculate_rmse()

    # 统计结果
    your_rmse_list = results["your_model"]
    other_rmse_list = results["other_model"]
    valid_nums = results["valid_nums"]

    # 计算平均值
    if len(your_rmse_list) > 0:
        your_rmse_mean = np.mean(your_rmse_list)
        other_rmse_mean = np.mean(other_rmse_list)

        # 打印汇总结果
        print("\n" + "=" * 80)
        print(
            f"有效计算数量：{len(valid_nums)} 张（编号 {min(valid_nums) if valid_nums else '无'} - {max(valid_nums) if valid_nums else '无'}）")
        print(f"你的模型 RMSE 平均值：{your_rmse_mean:.6f}")
        print(f"其他模型 RMSE 平均值：{other_rmse_mean:.6f}")

        # 对比结果
        if your_rmse_mean < other_rmse_mean:
            improvement = (other_rmse_mean - your_rmse_mean) / other_rmse_mean * 100
            print(f"\n✅ 你的模型更优！RMSE 降低了 {improvement:.2f}%")
        elif your_rmse_mean > other_rmse_mean:
            degradation = (your_rmse_mean - other_rmse_mean) / other_rmse_mean * 100
            print(f"\n❌ 其他模型更优！你的模型 RMSE 升高了 {degradation:.2f}%")
        else:
            print(f"\n🟡 两个模型 RMSE 完全相同")

        # 可选：打印详细统计
        print("\n详细统计：")
        print(
            f"你的模型 RMSE 最小值：{np.min(your_rmse_list):.6f} | 最大值：{np.max(your_rmse_list):.6f} | 标准差：{np.std(your_rmse_list):.6f}")
        print(
            f"其他模型 RMSE 最小值：{np.min(other_rmse_list):.6f} | 最大值：{np.max(other_rmse_list):.6f} | 标准差：{np.std(other_rmse_list):.6f}")
    else:
        print("\n⚠️  无有效数据计算RMSE，请检查文件路径和编号！")
