from typing import Tuple
from torch import Tensor

import torch
import torch.nn as nn

from .baseline import ResNet18, ResNet34
from .unet import MiniUNet

__all__ = ['MultiLabelNet18', 'MultiLabelNet34',
           'DensityNet18', 'DensityNet43', 'DensityWithMultiLabelNet18']


class MultiLabelNet18(ResNet18):
    def __init__(self, in_planes: int, out_dim_1st: int, out_dim_2nd: int, out_dim: int) -> None:
        super(MultiLabelNet18, self).__init__(in_planes, out_dim)
        self.fc_1st = nn.Linear(128, out_dim_1st)
        self.fc_2nd = nn.Linear(256, out_dim_2nd)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        out = self.maxpool(self.relu(self.bn1(self.conv1(x))))

        out = self.layer1(out)
        out = self.layer2(out)
        out_1st = self.avgpool(out)
        out_1st = torch.flatten(out_1st, 1)
        out_1st = self.fc_1st(out_1st)

        out = self.layer3(out)
        out_2nd = self.avgpool(out)
        out_2nd = torch.flatten(out_2nd, 1)
        out_2nd = self.fc_2nd(out_2nd)

        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out_1st, out_2nd, out


class MultiLabelNet34(ResNet34):
    def __init__(self, in_planes: int, out_dim_1st: int, out_dim_2nd: int, out_dim: int) -> None:
        super(MultiLabelNet34, self).__init__(in_planes, out_dim)
        self.fc_1st = nn.Linear(128, out_dim_1st)
        self.fc_2nd = nn.Linear(256, out_dim_2nd)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        out = self.maxpool(self.relu(self.bn1(self.conv1(x))))

        out = self.layer1(out)
        out = self.layer2(out)
        out_1st = self.avgpool(out)
        out_1st = torch.flatten(out_1st, 1)
        out_1st = self.fc_1st(out_1st)

        out = self.layer3(out)
        out_2nd = self.avgpool(out)
        out_2nd = torch.flatten(out_2nd, 1)
        out_2nd = self.fc_2nd(out_2nd)

        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out_1st, out_2nd, out


class DensityNet18(nn.Module):
    def __init__(self, out_dim: int) -> None:
        super(DensityNet18, self).__init__()
        self.density_net = MiniUNet(3, 1, True)
        self.classifier = ResNet18(1, out_dim)

    def forward(self, image: Tensor) -> Tuple[Tensor, Tensor]:
        density_out = self.density_net(image)

        out = self.classifier(density_out)

        return density_out, out


class DensityNet43(nn.Module):
    def __init__(self, out_dim: int) -> None:
        super(DensityNet43, self).__init__()
        self.density_net = MiniUNet(3, 1, True)
        self.classifier = ResNet34(1, out_dim)

    def forward(self, image: Tensor) -> Tuple[Tensor, Tensor]:
        density_out = self.density_net(image)

        out = self.classifier(density_out)

        return density_out, out


class DensityWithMultiLabelNet18(nn.Module):
    def __init__(self, out_dim_1st: int, out_dim_2nd: int, out_dim: int) -> None:
        super(DensityWithMultiLabelNet18, self).__init__()
        self.density_net = MiniUNet(3, 1, True)
        self.classifier = MultiLabelNet18(1, out_dim_1st, out_dim_2nd, out_dim)

    def forward(self, image: Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        density_out = self.density_net(image)

        out_1st, out_2nd, out = self.classifier(density_out)

        return density_out, (out_1st, out_2nd, out)
