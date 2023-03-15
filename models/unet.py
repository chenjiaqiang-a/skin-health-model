from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['MiniUNet', 'UNet']


class DoubleConv(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, mid_channels: int = None) -> None:
        super().__init__()
        if not mid_channels:
            mid_channels = out_planes
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_planes, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_planes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_planes: int, out_planes: int) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_planes, out_planes)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, bilinear: bool = True) -> None:
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_planes, out_planes, in_planes // 2)
        else:
            self.up = nn.ConvTranspose2d(in_planes, in_planes // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_planes, out_planes)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x1 = self.up(x1)

        # to ensure the shape of x1 and x2 are the same.
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_planes: int, out_planes: int) -> None:
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class MiniUNet(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, bilinear: bool = False) -> None:
        super(MiniUNet, self).__init__()
        factor = 2 if bilinear else 1
        self.inc = DoubleConv(in_planes, 8)
        self.down1 = Down(8, 16)
        self.down2 = Down(16, 32)
        self.down3 = Down(32, 64)
        self.down4 = Down(64, 128 // factor)
        self.up1 = Up(128, 64 // factor, bilinear)
        self.up2 = Up(64, 32 // factor, bilinear)
        self.up3 = Up(32, 16, bilinear)
        self.outc = OutConv(16, out_planes)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        out = self.up1(x5, x4)
        out = self.up2(out, x3)
        out = self.up3(out, x2)
        out = self.outc(out)
        return out


class UNet(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, bilinear: bool = False) -> None:
        super(UNet, self).__init__()

        self.inc = DoubleConv(in_planes, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, out_planes)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
