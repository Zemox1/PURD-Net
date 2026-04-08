import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2
from .layers import CombinedUpsample


class Decoder_A(nn.Module):
    def __init__(
            self,
            in_channels=1280,
            out_channels=3,
            decoder_width=0.6,


    ) -> None:
        super(Decoder_A, self).__init__()

        decoder_channels = int(in_channels * decoder_width)


        self.conv1x1 = nn.Conv2d(
            in_channels, decoder_channels, kernel_size=1, stride=1, padding=0  #
        )

        # 上采样层（保持原结构）
        self.up0 = CombinedUpsample(decoder_channels + 320, decoder_channels // 2)  # 修正通道数计算
        self.up1 = CombinedUpsample(decoder_channels // 2 + 160, decoder_channels // 2)
        self.up2 = CombinedUpsample(decoder_channels // 2 + 64, decoder_channels // 4)
        self.up3 = CombinedUpsample(decoder_channels // 4 + 32, decoder_channels // 8)
        self.up4 = CombinedUpsample(decoder_channels // 8 + 24, decoder_channels // 8)
        self.up5 = CombinedUpsample(decoder_channels // 8 + 16, 48)


        self.up6 = CombinedUpsample(48 + 12, 32)
        self.conv2d = nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1)

        # 添加激活函数，将输出约束到[0,1]范围（适用于RGB图像）
        self.output_activation = nn.Sigmoid()
        self.scale = nn.Parameter(torch.tensor(1.0))



    def forward(self, features):
        # 提取跳跃连接特征（确保尺寸匹配）
        skip0 = features[2]  # 16 ×240×320
        skip1 = features[4]  # 24 ×120×160
        skip2 = features[6]  # 32 ×60×80
        skip3 = features[9]  # 64 ×30×40
        skip4 = features[15]  # 160 ×15×20
        skip5 = features[18]  # 320 ×15×20
        out = features[19]  # 1280×15×20

        # 特征处理与上采样
        out = self.conv1x1(out)  # 768×15×20（修正padding后尺寸更准确）
        out = self.up0(out, skip5)  # 384×15×20
        out = self.up1(out, skip4)  # 384×15×20
        out = self.up2(out, skip3)  # 192×30×40
        out = self.up3(out, skip2)  # 96×60×80
        out = self.up4(out, skip1)  # 96×120×160
        out = self.up5(out, skip0)  # 48×240×320
        rgb_skip = features[0]  # 假设为3×480×640
        out = self.up6(out, rgb_skip)  # 32×480×640
        out = self.conv2d(out)  # 3×480×640
        out = out * self.scale

        # 激活函数将输出约束到[0,1]
        out = self.output_activation(out)  # 先保证非负

        return out
