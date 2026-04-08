import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as torch_models

class SILogLoss(nn.Module):
    """Scale invariant logarithmic loss.
    
    Inspired by https://arxiv.org/abs/1406.2283 \\
    and https://arxiv.org/pdf/2011.14141.pdf"""

    def __init__(self, correction=1.0, scaling=10.0, eps=1e-10) -> None:
        """correction: in range [0,1], where 0 results in a loss equivalent to RMSE in log space
        and 1 results in RMSE in log space with scale invariance."""

        super(SILogLoss, self).__init__()
        self.name = "SILog"

        self.eps = eps  # avoid log(0)
        self.correction = correction
        self.scaling = scaling

    def forward(self, prediction, target):



        # elementwise log difference
        d = torch.log(prediction + self.eps) - torch.log(target + self.eps)

        # loss
        loss = torch.mean(torch.pow(d, 2)) - self.correction * torch.pow(
            torch.mean(d), 2
        )

        # alternative implementation used by UDepth and AdaBins using "Bessels Correction"
        # (torch.var is using bessels correction by default, see arg "unbiased")
        # loss2 = torch.var(d) + 0.15 * torch.pow(torch.mean(d), 2)

        return self.scaling * torch.sqrt(loss)


class ChamferDistanceLoss(nn.Module):
    """Chamfer Distance Loss.
    Target images and bin centers are normalized by deviding by corresponding max.

    Inspired by https://arxiv.org/abs/1612.00603"""

    def __init__(self, scale_invariant=True) -> None:
        super(ChamferDistanceLoss, self).__init__()

        self.name = "ChamferDistance"

        self.scale_invariant = scale_invariant

    def onedirectional_dist(self, a, b):

        # manually assign memory and use expand instead of repeat to minimize memory usage
        distances = torch.empty(a.size(0), a.size(1), b.size(1))  # 1xAxB
        distances[...] = a.unsqueeze(-1).expand(
            a.size(0), a.size(1), b.size(1)
        ) - b.unsqueeze(-1).expand(b.size(0), b.size(1), a.size(1)).permute(0, 2, 1)

        # squared distance matrix
        distances = distances.pow(2)

        # find nearest neighbor distance for each point in a, NxA
        nn_squared_distances = distances.amin(dim=2)

        # summed distance
        sum = nn_squared_distances.sum()

        # per point average
        sum /= a.size(1)

        return sum

    def forward(self, target, bin_centers):
        # target shape: Nx1xHxW
        # bin_centers shape: NxB

        # normalize, global scale should have no effect
        if self.scale_invariant:
            target_max = target.amax(dim=(2, 3))[..., None, None]
            # norm target and bins by target max
            target_n = target / target_max  # [..., None, None, None]
            bin_centers_n = bin_centers / target_max[..., 0, 0]
        else:
            target_n = target.clone()
            bin_centers_n = bin_centers.clone()

        # doing imgs sequential takes slightly longer but drastically lowers memory usage
        n_batch = target.size(0)
        bidirectional_dist = torch.zeros(1).to(target.device)
        for i in range(n_batch):

            target_depths = target_n[i, ...].flatten().unsqueeze(0)

            bidirectional_dist += self.onedirectional_dist(
                bin_centers_n[i].unsqueeze(0), target_depths
            ) + self.onedirectional_dist(target_depths, bin_centers_n[i].unsqueeze(0))

        # mean over all batches
        bidirectional_dist = bidirectional_dist / n_batch

        return bidirectional_dist * 10.0


class RMSELoss(nn.Module):
    """Root Mean Squared Error (RMSE)"""

    def __init__(self) -> None:
        super(RMSELoss, self).__init__()

        self.name = "RMSELoss"

        self.mse_loss = nn.MSELoss()

    def forward(self, prediction, target):

        loss = torch.sqrt(self.mse_loss(prediction, target))

        return loss


class MARELoss(nn.Module):
    """Mean Absolute Relative Error (MARE)"""

    def __init__(self) -> None:
        super(MARELoss, self).__init__()

        self.name = "MARELoss"

    def forward(self, prediction, target, mask=None):

        # apply mask
        if mask is not None:
            prediction = prediction[mask]
            target = target[mask]

        loss = torch.mean(torch.abs((prediction - target) / target))

        return loss


def ssim_loss(img1, img2, C1=1e-4, C2=9e-4):
    mu1 = F.avg_pool2d(img1, 3, 1, 1)
    mu2 = F.avg_pool2d(img2, 3, 1, 1)

    sigma1 = F.avg_pool2d(img1 * img1, 3, 1, 1) - mu1 * mu1
    sigma2 = F.avg_pool2d(img2 * img2, 3, 1, 1) - mu2 * mu2
    sigma12 = F.avg_pool2d(img1 * img2, 3, 1, 1) - mu1 * mu2

    # 计算SSIM
    ssim_n = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    ssim_d = (mu1 * mu1 + mu2 * mu2 + C1) * (sigma1 + sigma2 + C2)

    ssim_map = ssim_n / ssim_d
    return 1 - ssim_map.mean()  # 返回1 - SSIM值


class PerceptualLoss(nn.Module):
    """
    感知损失：在预训练CNN特征空间计算差异，适配大气散射图估计
    特点：1. 保留空间特征（不用全连接层）；2. 平衡浅层细节与深层结构；3. 支持单通道散射图
    """

    def __init__(self,
                 layers=[2, 7, 12, 21],  # VGG19的特征层（浅层到深层）
                 weights=[1.0, 0.8, 0.5, 0.3],  # 浅层权重更高（散射细节更重要）
                 backbone='vgg19',
                 device='cuda'):
        super().__init__()
        self.name = "PerceptualLoss"
        self.device = device
        self.weights = torch.tensor(weights, device=device) / sum(weights)  # 归一化权重

        # 加载预训练backbone并冻结参数
        if backbone == 'vgg19':
            self.backbone = torch_models.vgg19(pretrained=True).features.to(device)
        elif backbone == 'vgg16':
            self.backbone = torch_models.vgg16(pretrained=True).features.to(device)
        else:
            raise ValueError("backbone仅支持vgg16/vgg19")

        # 提取指定层的特征提取器
        self.feature_extractors = nn.ModuleList()
        for layer in layers:
            extractor = nn.Sequential(*list(self.backbone[:layer + 1])).to(device)
            for param in extractor.parameters():
                param.requires_grad = False  # 冻结backbone
            self.feature_extractors.append(extractor)

        # VGG输入归一化参数（ImageNet均值和方差）
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def preprocess(self, x):
        """预处理输入：支持单通道（散射图）或三通道（去散射图像），转为VGG输入格式"""
        if x.dim() == 3:
            x = x.unsqueeze(0)  # 加batch维度：[H,W] -> [1,1,H,W]
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)  # 单通道转三通道（散射图复制为3通道匹配VGG）
        x = (x - self.mean) / self.std  # 归一化到VGG训练分布
        return x

    def forward(self, pred, target):
        """
        Args:
            pred: 预测图（可以是散射图或去散射后的清晰图，shape: [B,1,H,W]或[B,3,H,W]）
            target: 真实图（同pred格式）
        Returns:
            加权后的感知损失
        """
        # 预处理输入
        pred = self.preprocess(pred)
        target = self.preprocess(target)

        total_loss = 0.0
        for i, extractor in enumerate(self.feature_extractors):
            # 提取特征
            pred_feat = extractor(pred)
            target_feat = extractor(target)
            # 计算特征差异（用L1损失更抗噪，适合散射图的高频噪声）
            loss = F.l1_loss(pred_feat, target_feat)
            # 加权累加（浅层细节权重更高）
            total_loss += loss * self.weights[i]

        return total_loss
    

class LabColorLoss(nn.Module):
    def __init__(self, weight_L=1.0, weight_ab=5.0):
        super().__init__()
        self.weight_L = weight_L
        self.weight_ab = weight_ab
        
    def rgb_to_xyz(self, rgb):
        """RGB转XYZ颜色空间"""
        mask = rgb > 0.04045
        rgb = torch.where(mask, torch.pow((rgb + 0.055) / 1.055, 2.4), rgb / 12.92)
        
        # 转换矩阵
        transform = torch.tensor([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ]).to(rgb.device)
        
        # 应用转换矩阵
        xyz = torch.einsum('ij,bjhw->bihw', transform, rgb)
        return xyz
    
    def xyz_to_lab(self, xyz):
        """XYZ转Lab颜色空间"""
        # D65白点参考值
        ref_white = torch.tensor([0.95047, 1.0, 1.08883]).to(xyz.device).view(1, 3, 1, 1)
        
        # 归一化
        xyz_normalized = xyz / ref_white
        
        # 非线性转换
        mask = xyz_normalized > 0.008856
        xyz_normalized = torch.where(
            mask, 
            torch.pow(xyz_normalized, 1/3), 
            7.787 * xyz_normalized + 16/116
        )
        
        # 计算Lab分量
        L = torch.where(
            xyz[:, 1:2, :, :] / ref_white[:, 1:2, :, :] > 0.008856,
            116 * torch.pow(xyz[:, 1:2, :, :] / ref_white[:, 1:2, :, :], 1/3) - 16,
            903.3 * xyz[:, 1:2, :, :] / ref_white[:, 1:2, :, :]
        )
        
        a = 500 * (xyz_normalized[:, 0:1, :, :] - xyz_normalized[:, 1:2, :, :])
        b = 200 * (xyz_normalized[:, 1:2, :, :] - xyz_normalized[:, 2:3, :, :])
        
        lab = torch.cat([L, a, b], dim=1)
        return lab
    
    def rgb_to_lab(self, rgb):
        """RGB转Lab颜色空间"""
        xyz = self.rgb_to_xyz(rgb)
        lab = self.xyz_to_lab(xyz)
        return lab
    
    def forward(self, pred, target):
        # 确保输入在[0,1]范围内
        pred = torch.clamp(pred, 0, 1)
        target = torch.clamp(target, 0, 1)
        
        # 转换为Lab颜色空间
        pred_lab = self.rgb_to_lab(pred)
        target_lab = self.rgb_to_lab(target)
        
        # 分离L和ab通道
        pred_L, pred_ab = pred_lab[:, 0:1, :, :], pred_lab[:, 1:3, :, :]
        target_L, target_ab = target_lab[:, 0:1, :, :], target_lab[:, 1:3, :, :]
        
        # 计算损失
        L_loss = nn.functional.l1_loss(pred_L, target_L)
        ab_loss = nn.functional.l1_loss(pred_ab, target_ab)
        
        return self.weight_L * L_loss + self.weight_ab * ab_loss
    
def masked_l2_loss(pred, target, weights):
    # 1. 计算平方误差
    squared_error = (pred - target) ** 2
    
    # 2. 如果有mask则加权，否则直接平均
    if weights is not None:
        # 加权平均
        weighted_mse = (squared_error * weights).mean()
    else:
        # 普通平均
        weighted_mse = squared_error.mean()
    
    # 3. 开方得到RMSE
    return torch.sqrt(weighted_mse)


def test_chamfer():

    # batch size
    n_batch = 4

    # bins
    n_bins = 80
    bin_edges = torch.arange(0.0, 1.0, 1.0 / n_bins)
    bin_edges = bin_edges.unsqueeze(0).repeat(n_batch, 1)
    bin_centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
    print(f"bin center shape: {bin_centers.shape}")

    # random image
    img = torch.rand(n_batch, 1, 240, 320)

    # loss
    lossfunc = ChamferDistanceLoss(scale_invariant=True)

    print("Testing scale invariance ...")
    scale = 2.0
    assert lossfunc(img, bin_centers) == lossfunc(scale * img, scale * bin_centers)

    print(f"img [0,1], bins [0,1], loss: {lossfunc(img, bin_centers)}")

    # mask
    mask = img.lt(0.8)
    # change invalid part of img to 1.0
    img[~mask] = img[mask].max()

    print("Testing scale invariance with mask...")
    scale = 2.0
    assert lossfunc(img, bin_centers, mask) == lossfunc(
        scale * img, scale * bin_centers, mask
    )

    print("Test masking option ...")
    assert lossfunc(img, bin_centers) > lossfunc(img, bin_centers, mask)

    print("Tests succeeded.")


if __name__ == "__main__":
    test_chamfer()
