from typing import Tuple
from torch import Tensor

import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet50

from .unet import MiniUNet

__all__ = ['ResNet18', 'ResNet50',
           'DensityNet18', 'DensityNet50']


class ResNet18(nn.Module):
    def __init__(self, out_dim: int, in_channels: int = None) -> None:
        super(ResNet18, self).__init__()
        self.classifier = resnet18(pretrained=True)
        self.classifier.fc = nn.Linear(512, out_dim)
        if in_channels is not None:
            self.classifier.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2,
                                              padding=3, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(x)

    def __str__(self):
        return "<ResNet18>"


class ResNet50(nn.Module):
    def __init__(self, out_dim: int, in_channels: int = None) -> None:
        super(ResNet50, self).__init__()
        self.classifier = resnet50(pretrained=True)
        self.classifier.fc = nn.Linear(2048, out_dim)
        if in_channels is not None:
            self.classifier.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2,
                                              padding=3, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(x)

    def __str__(self):
        return "<ResNet50>"


class DensityNet18(nn.Module):
    def __init__(self, out_dim: int, in_channels: int = 3) -> None:
        super(DensityNet18, self).__init__()
        self.density_net = MiniUNet(in_channels, 1, True)
        self.classifier = ResNet18(out_dim, 1)

    def forward(self, image: Tensor) -> Tuple[Tensor, Tensor]:
        density_out = self.density_net(image)
        out = self.classifier(density_out)

        return density_out, out

    def __str__(self):
        return "<DensityNet18>"


class DensityNet50(nn.Module):
    def __init__(self, out_dim: int, in_channels: int = 3) -> None:
        super(DensityNet50, self).__init__()
        self.density_net = MiniUNet(in_channels, 1, True)
        self.classifier = ResNet50(out_dim, 1)

    def forward(self, image: Tensor) -> Tuple[Tensor, Tensor]:
        density_out = self.density_net(image)
        out = self.classifier(density_out)

        return density_out, out

    def __str__(self):
        return "<DensityNet50>"
