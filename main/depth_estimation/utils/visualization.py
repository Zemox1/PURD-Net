# import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.types import Device
from torchvision.utils import make_grid


def get_bin_centers_img(bin_edges, target,  device="cpu"):
    """Get visualization for usage on e.g. tensorboard which shows distribution
    of target depths and predicted bins."""

    # target shapes
    n_batch = target.size(0)
    n_edges = bin_edges.size(1)
    n_bins = n_edges - 1
    height = target.size(2)
    width = target.size(3)

    # get target bin edges by sampling the sorged target imgs at their quantiles
    target_bin_edges = torch.empty(n_batch, n_edges).to(device)
    for i in range(n_batch):

        target_sorted, _ = target[i].view(-1).sort()
        step = target_sorted.size(0) / n_edges
        target_bin_edge_idcs = (torch.arange(n_edges) * step).long()
        target_bin_edges[i, ...] = target_sorted[target_bin_edge_idcs]

    # norm bin edges to [0,1]
    max_edges = bin_edges.amax(dim=1)
    max_target_edges = target_bin_edges.amax(dim=1)
    max = torch.stack([max_edges, max_target_edges]).amax(dim=0).unsqueeze(-1)  # Nx1
    bin_edges_normed = bin_edges / max
    target_bin_edges_normed = target_bin_edges / max

    # initialize out img and lines
    bin_edges_img = torch.zeros(n_batch, 1, height, width).to(device)
    bin_edges_img_line = torch.ones(n_batch, 1, 1, width).to(device) * 0.5
    target_bin_edges_img_line = torch.ones(n_batch, 1, 1, width).to(device) * 0.5

    n_bins = bin_edges_normed.size(1) - 1
    for i in range(n_batch):

        # draw bins
        black_white = True  # alternating black/white color for good visibility
        for j in range(n_bins):

            # get bin edges
            edge_start = (bin_edges_normed[i, j] * (width - 1)).int().item()
            edge_end = (bin_edges_normed[i, j + 1] * (width - 1)).int().item()
            target_edge_start = (
                (target_bin_edges_normed[i, j] * (width - 1)).int().item()
            )
            target_edge_end = (
                (target_bin_edges_normed[i, j + 1] * (width - 1)).int().item()
            )

            # draw lines
            bin_edges_img_line[i, 0, 0, edge_start:edge_end] = float(black_white)
            target_bin_edges_img_line[
                i, 0, 0, target_edge_start:target_edge_end
            ] = float(black_white)

            # alternate color
            black_white = not black_white

    # expand lines to multiple img rows to create full img
    bin_edges_img[:, :, : int(height / 2), :] = target_bin_edges_img_line.expand(
        n_batch, 1, int(height / 2), width
    )
    bin_edges_img[:, :, int(height / 2) :, :] = bin_edges_img_line.expand(
        n_batch, 1, int(height / 2), width
    )

    return bin_edges_img



def gray_to_magma(gray, colormap="magma", device=Device, normalize=True):
    """
    仿照gray_to_inferno实现：将深度图转换为MAGMA配色
    核心修改：配色替换为magma，其余逻辑完全对齐
    新增：处理0值和异常值，避免右侧出现大片黑色
    """
    # 使用MAGMA配色（核心替换点）
    colormap = plt.get_cmap('magma')

    # 转换为CPU并拆分批次（兼容批量输入）
    gray_imgs = [img.cpu().detach() for img in gray.unbind(dim=0)]

    processed = []
    for img in gray_imgs:
        # 去除通道维度，处理0值和异常值（NaN/Inf）
        img = img.squeeze()
        valid_mask = (img > 0) & (~torch.isnan(img)) & (~torch.isinf(img))  # 增强：额外处理Inf

        if valid_mask.any():
            # 用有效区域最小值替换无效值（避免0值/异常值导致的黑色块）
            img[~valid_mask] = img[valid_mask].min()

            # 归一化（保留原有逻辑，增强数值鲁棒性）
            if normalize:
                min_val = img.min()
                max_val = img.max()
                if max_val > min_val + 1e-8:  # 避免除零
                    img = (img - min_val) / (max_val - min_val)
        processed.append(img)

    # 应用magma配色并转换格式（和原函数逻辑完全一致）
    magmas = [colormap(img)[..., :3] for img in processed]  # 取RGB通道，丢弃Alpha
    magmas = np.stack(magmas, axis=0)  # 恢复批次维度
    # 转换为Tensor并调整维度：(B, H, W, 3) → (B, 3, H, W)，适配PyTorch格式
    magmas = torch.from_numpy(magmas).permute(0, 3, 1, 2).to(device)

    return magmas


def gray_to_inferno(gray,colormap="inferno", device=Device, normalize=True):
    """
    简单修改版：将深度图转换为INFERNO配色
    新增：处理0值和异常值，避免右侧出现大片黑色
    """
    # 使用INFERNO配色
    colormap = plt.get_cmap('inferno')

    # 转换为CPU并拆分批次
    gray_imgs = [img.cpu().detach() for img in gray.unbind(dim=0)]

    processed = []
    for img in gray_imgs:
        # 去除通道维度，处理0值和异常值
        img = img.squeeze()
        valid_mask = (img > 0) & (~torch.isnan(img))

        if valid_mask.any():
            # 用有效区域最小值替换无效值（避免0值导致的黑色）
            img[~valid_mask] = img[valid_mask].min()

            # 归一化（保留原有逻辑并增强鲁棒性）
            if normalize:
                min_val = img.min()
                max_val = img.max()
                if max_val > min_val + 1e-8:
                    img = (img - min_val) / (max_val - min_val)
        processed.append(img)

    # 应用配色并转换格式
    infernos = [colormap(img)[..., :3] for img in processed]
    infernos = np.stack(infernos, axis=0)
    infernos = torch.from_numpy(infernos).permute(0, 3, 1, 2).to(device)

    return infernos


def get_tensorboard_grids(X, y, pred,  bin_edges, device="cpu"):
    """Generates tensorboard grids for tensorboard summary writer.

    Inputs:
    - X: RGB input [Nx3xHxW]
    - y: ground truth depth [Nx1xHxW]
    - prior: prior parametrization [Nx2xHxW]
    - pred: prediction [Nx1xHxW]
    - mask: mask for valid depths [Nx1xHxW]
    - bin_edges: [Nxn_bins]
    """

    # error
    error = torch.abs(y - pred)


    # target parametrization


    # resize rgb
    rgb_resized = torch.nn.functional.interpolate(
        X, size=[pred.size(2), pred.size(3)], mode="bilinear", align_corners=True
    )

    # get bin center visualization
    bin_centers_img = get_bin_centers_img(bin_edges, y, device=device)

    # get heatmaps
    y_heatmap = gray_to_inferno(y, device=device)
    pred_heatmap = gray_to_inferno(pred, device=device)

    error_heatmap = gray_to_inferno(error, colormap="inferno", device=device)

    bin_centers_heatmap = gray_to_inferno(
        bin_centers_img, colormap="inferno", device=device
    )

    # grids
    nrow = X.size(0)
    rgb_target_pred_error_grid = make_grid(
        torch.cat(
            (rgb_resized, y_heatmap, pred_heatmap, error_heatmap, bin_centers_heatmap),
            dim=0,
        ),
        nrow=nrow,
    )
    prior_parametrization_grid = make_grid(
        torch.cat((y_heatmap, ), dim=0),
        nrow=nrow,
    )

    return rgb_target_pred_error_grid, prior_parametrization_grid
