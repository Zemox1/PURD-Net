import torch
import torch.nn as nn
import torch.nn.functional as F



class Decoder_A_Simple(nn.Module):
    def __init__(
            self,
            in_channels=1280,
            out_channels=3,
            decoder_width=0.5,

    ) -> None:
        super(Decoder_A_Simple, self).__init__()

        decoder_channels = int(in_channels * decoder_width)

        # 1x1卷积调整通道数
        self.conv1x1 = nn.Conv2d(
            in_channels, decoder_channels, kernel_size=1, stride=1, padding=0
        )

        # 直接上采样到目标尺寸，不使用CombinedUpsample避免尺寸问题
        self.upsample = nn.Sequential(
            # 从15×20开始上采样
            nn.Conv2d(decoder_channels, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 30×40

            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 60×80

            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 120×160

            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 240×320

            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 480×640

            nn.Conv2d(32, out_channels, 3, padding=1),
        )

        # 输出激活
        self.output_activation = nn.Sigmoid()
        self.scale = nn.Parameter(torch.tensor(1.0))

        # 高斯模糊
        self.gaussian_blur = GaussianBlur(kernel_size=7, sigma=1.0)

    def forward(self, features):
        # 只使用最高层特征
        out = features[19]  # 1280×15×20

        # 简化的处理流程
        out = self.conv1x1(out)  # decoder_channels×15×20

        # 直接上采样到目标尺寸
        out = self.upsample(out)  # out_channels×480×640

        out = out * self.scale
        out = self.output_activation(out)
        out = self.gaussian_blur(out)

        return out

class GaussianBlur(nn.Module):
    def __init__(self, kernel_size=15, sigma=3.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma

    def forward(self, x):
        return self.gaussian_blur(x, self.kernel_size, self.sigma)

    def gaussian_blur(self, x, kernel_size, sigma):
        # 创建高斯核
        kernel = self.get_gaussian_kernel(kernel_size, sigma).to(x.device)
        kernel = kernel.view(1, 1, kernel_size, 1) * kernel.view(1, 1, 1, kernel_size)
        kernel = kernel.repeat(x.size(1), 1, 1, 1)

        padding = kernel_size // 2
        blurred = F.conv2d(x, kernel, padding=padding, groups=x.size(1))
        return blurred

    def get_gaussian_kernel(self, size, sigma):
        coords = torch.arange(size).float() - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return g