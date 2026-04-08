import torch.nn as nn
from torchvision.models import mobilenet_v2

from .layers import CombinedUpsample

import torch.nn as nn
from torchvision.models import mobilenet_v2


class Encoder(nn.Module):
    def __init__(self) -> None:
        super(Encoder, self).__init__()

        # 加载预训练的MobileNetV2
        self.original_model = mobilenet_v2(pretrained=True)

        # 修改第一个卷积层的输入通道：从3改为12
        # MobileNetV2的第一层是features[0]中的conv.conv（具体结构需根据源码确认）
        # 原第一层结构：Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        first_conv = self.original_model.features[0][0]  # 获取第一层卷积
        # 创建新的12输入通道卷积层，复用原权重的前3通道（或随机初始化）
        new_first_conv = nn.Conv2d(
            in_channels=12,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None
        )

        # 初始化新卷积层的权重（可选策略）
        if first_conv.weight is not None:
                original_weight = first_conv.weight.data  # [out_channels, 3, k, k]
                # 将原始3通道权重复制4次来填充12个输入通道
                new_weight = original_weight.repeat(1, 4, 1, 1)  # [out_channels, 12, k, k]
                new_first_conv.weight.data = new_weight
           

        # 替换第一层卷积
        self.original_model.features[0][0] = new_first_conv

    def forward(self, x):
        features = []
        features.append(x)  # 保存输入作为第一个特征（skip connection用）
        # 逐层提取特征
        for submodule in self.original_model.features:
            x = submodule(x)
            features.append(x)

        return features


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels=1280,
        out_channels=3,

        decoder_width=0.6,
        single_channel_output=False,
    ) -> None:
        super(Decoder, self).__init__()

        decoder_channels = int(in_channels * decoder_width)

        # 1x1 convolution to reduce/expand channels
        self.conv1x1 = nn.Conv2d(
            in_channels, decoder_channels, kernel_size=1, stride=1, padding=1
        )

        # upsampling layers
        # in_channels equals current channels + concat channels + prior channels
        self.up0 = CombinedUpsample(
            decoder_channels // 1 + 320 , decoder_channels // 2
        )
        self.up1 = CombinedUpsample(
            decoder_channels // 2 + 160 , decoder_channels // 2
        )
        self.up2 = CombinedUpsample(
            decoder_channels // 2 + 64 , decoder_channels // 4
        )
        self.up3 = CombinedUpsample(
            decoder_channels // 4 + 32 , decoder_channels // 8
        )
        self.up4 = CombinedUpsample(
            decoder_channels // 8 + 24 , decoder_channels // 8
        )
        self.up5 = CombinedUpsample(
            decoder_channels // 8 + 16 , 48
        )
        self.up6 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # optional: 3x3 convolution, 1 channel output
        # for estimating depth directly from encoder-decoder
        self.to_rgb = nn.Conv2d(48, out_channels, kernel_size=1, bias=True)

        self.single_channel_output = single_channel_output
        if self.single_channel_output:
            self.conv3x3 = nn.Conv2d(
                decoder_channels // 16 ,
                1,
                kernel_size=3,
                stride=1,
                padding=1,
            )

    def forward(self, features):

        # use subset of intermediate features as skip connections
        skip0 = features[2]  # size 16 x 240 x 320
        skip1 = features[4]  # size 24 x 120 x 160
        skip2 = features[6]  # size 32 x 60 x 80
        skip3 = features[9]  # size 64 x 30 x 40
        skip4 = features[15]  # size 160 x 15 x 20
        skip5 = features[18]  # size 320 x 15 x 20
        out = features[19]  # size 1280 x 15 x 20

        # convolve input to match decoder channnels c_in -> c
        out = self.conv1x1(out)  # size c x 15 x 20

        # upsample together with skip connections and depth prior
        out = self.up0(out, skip5)  # size c//2 x 15 x 20
        out = self.up1(out, skip4)  # size c//2 x 15 x 20
        out = self.up2(out, skip3)  # size c//4 x 30 x 40
        out = self.up3(out, skip2)  # size c//8 x 30 x 40
        out = self.up4(out, skip1)  # size c//8 x 120 x 160
        out = self.up5(out, skip0)  # size c_out x 240 x 320
        out = self.up6(out)  # 48×480×640
        out = self.to_rgb(out)
        # optional: final convolution to achieve single channel
        if self.single_channel_output:
            out = self.conv3x3(out)

        return out
