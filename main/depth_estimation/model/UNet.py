import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """双卷积块"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),  # 添加BN，有助于深度图训练
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """深度图refine网络
    输入: 深度图(1通道) + RGB图像(3通道) = 4通道
    输出: 精细化深度图(1通道)
    """

    def __init__(self, in_channels=4, out_channels=1,device=None):
        super().__init__()
        self.device=device
        if device is not None:
            self.to(device)
        # 编码器 (下采样路径)
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        # 下采样
        self.pool = nn.MaxPool2d(2)

        # 瓶颈层
        self.bottleneck = DoubleConv(512, 1024)

        # 解码器 (上采样路径)
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)  # 1024 = 512(上采样) + 512(跳连)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)  # 512 = 256(上采样) + 256(跳连)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)  # 256 = 128(上采样) + 128(跳连)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)  # 128 = 64(上采样) + 64(跳连)

        # 输出层
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

        # 可选的输出激活函数
        self.sigmoid = nn.Sigmoid()  # 如果深度值归一化到[0,1]
        # self.relu = nn.ReLU()        # 如果深度值为正值

    def forward(self, x):
        """
        前向传播
        输入: [batch, 4, 480, 640]   (1通道深度 + 3通道RGB)
        输出: [batch, 1, 480, 640]   (精细化深度图)
        """
        # 编码器路径
        e1 = self.enc1(x)  # [B, 64, 480, 640]
        e2 = self.enc2(self.pool(e1))  # [B, 128, 240, 320]
        e3 = self.enc3(self.pool(e2))  # [B, 256, 120, 160]
        e4 = self.enc4(self.pool(e3))  # [B, 512, 60, 80]

        # 瓶颈层
        bottleneck = self.bottleneck(self.pool(e4))  # [B, 1024, 30, 40]

        # 解码器路径 (包含跳连连接)
        d4 = self.up4(bottleneck)  # [B, 512, 60, 80]
        d4 = torch.cat([d4, e4], dim=1)  # 跳连连接
        d4 = self.dec4(d4)  # [B, 512, 60, 80]

        d3 = self.up3(d4)  # [B, 256, 120, 160]
        d3 = torch.cat([d3, e3], dim=1)  # 跳连连接
        d3 = self.dec3(d3)  # [B, 256, 120, 160]

        d2 = self.up2(d3)  # [B, 128, 240, 320]
        d2 = torch.cat([d2, e2], dim=1)  # 跳连连接
        d2 = self.dec2(d2)  # [B, 128, 240, 320]

        d1 = self.up1(d2)  # [B, 64, 480, 640]
        d1 = torch.cat([d1, e1], dim=1)  # 跳连连接
        d1 = self.dec1(d1)  # [B, 64, 480, 640]

        # 输出层
        output = self.out_conv(d1)  # [B, 1, 480, 640]

        # 根据深度图数值范围选择合适的激活函数
        output = self.sigmoid(output)  # 如果深度值在[0,1]
        # output = self.relu(output)     # 如果深度值>=0

        return output

    def forward_with_rgb_depth(self, depth, rgb):

        # 在通道维度拼接
        x = torch.cat([depth, rgb], dim=1)  # [batch, 4, H, W]
        return self.forward(x)


# 测试代码
if __name__ == "__main__":
    # 创建模型
    model = UNet(in_channels=4, out_channels=1)

    # 测试数据
    batch_size = 2
    depth_map = torch.randn(batch_size, 1, 480, 640)  # 深度图
    rgb_image = torch.randn(batch_size, 3, 480, 640)  # RGB图像

    print("=== 深度图Refine UNet测试 ===")
    print(f"输入深度图尺寸: {depth_map.shape}")
    print(f"输入RGB图像尺寸: {rgb_image.shape}")

    # 方法1: 直接拼接后输入
    x = torch.cat([depth_map, rgb_image], dim=1)
    output1 = model(x)
    print(f"\n方法1输出尺寸: {output1.shape}")

    # 方法2: 使用便捷接口
    output2 = model.forward_with_rgb_depth(depth_map, rgb_image)
    print(f"方法2输出尺寸: {output2.shape}")

    # 验证输出
    if output1.shape == (batch_size, 1, 480, 640):
        print("✓ 输出尺寸正确！")

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n模型总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
