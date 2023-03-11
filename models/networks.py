from typing import Tuple
from torch import Tensor

import torch
import torch.nn as nn

from .baseline import ResNet18, ResNet34
from .unet import MiniUNet

__all__ = ['MultiLabelNet18', 'MultiLabelNet34',
           'DensityNet18', 'DensityWithMultiLabelNet18']


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
        out_1st = self.fc_1st(out_1st)

        out = self.layer3(out)
        out_2nd = self.avgpool(out)
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
        out_1st = self.fc_1st(out_1st)

        out = self.layer3(out)
        out_2nd = self.avgpool(out)
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

    def forward(self, image: Tensor, density_gt: Tensor = None, d_mask: Tensor = None) -> Tuple[Tensor, Tensor]:
        density = self.density_net(image)

        if self.training:
            assert density_gt is not None and d_mask is not None,\
                "Please input density_gt and d_mask while training!"
            density[d_mask] = density_gt[d_mask]

        out = self.classifier(density)

        return density, out


class DensityWithMultiLabelNet18(nn.Module):
    def __init__(self, out_dim_1st: int, out_dim_2nd: int, out_dim: int) -> None:
        super(DensityWithMultiLabelNet18, self).__init__()
        self.density_net = MiniUNet(3, 1, True)
        self.classifier = MultiLabelNet18(1, out_dim_1st, out_dim_2nd, out_dim)

    def forward(
            self,
            image: Tensor,
            density_gt: Tensor = None,
            d_mask: Tensor = None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        density = self.density_net(image)

        if self.training:
            assert density_gt is not None and d_mask is not None, \
                "Please input density_gt and d_mask while training!"
            density[d_mask] = density_gt[d_mask]

        out_1st, out_2nd, out = self.classifier(density)

        return density, (out_1st, out_2nd, out)
