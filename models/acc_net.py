import torch
import torch.nn as nn
from torch import Tensor


def conv7x7(in_planes: int, out_planes: int, stride: int = 1, padding: int = 3) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=7,
                     stride=stride, padding=padding, bias=False)


def conv5x5(in_planes: int, out_planes: int, stride: int = 1, padding: int = 2) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=5,
                     stride=stride, padding=padding, bias=False)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, padding: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=padding, bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1,
                     stride=stride, bias=False)


class BasicBlock(nn.Module):
    def __init__(self,
                 in_planes: int,
                 planes: int,
                 stride: int = 1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = None
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                conv1x1(in_planes, planes, stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.shortcut is not None:
            identity = self.shortcut(x)

        out += identity
        out = self.relu(out)

        return out


class ACCNet(nn.Module):
    def __init__(self, out_dim: int, out_1st_dim: int = None, out_2nd_dim: int = None):
        super(ACCNet, self).__init__()
        self.shared_conv = nn.Sequential(
            conv7x7(3, 16),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            conv5x5(16, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.density_net = nn.Sequential(
            conv3x3(32, 16),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3(16, 8),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            conv1x1(8, 1),
        )
        self.conv = conv3x3(32, 63, stride=2)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, 2)
        self.layer3 = self._make_layer(256, 2, 2)
        self.layer4 = self._make_layer(512, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = None
        if out_1st_dim is not None:
            self.fc1 = nn.Linear(128, out_1st_dim)
        self.fc2 = None
        if out_2nd_dim is not None:
            self.fc2 = nn.Linear(256, out_2nd_dim)
        self.fc = nn.Linear(512, out_dim)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1):
        layers = [BasicBlock(self.in_planes, planes, stride)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor, density_gt: Tensor = None, d_mask: Tensor = None):
        feature = self.shared_conv(x)
        density_map = self.density_net(feature)

        density = density_map.clone()
        if self.training:
            assert density_gt is not None and d_mask is not None,\
                "Please input density_gt and d_mask while training!"
            density[d_mask] = density_gt[d_mask]

        feature = self.conv(feature)
        feature = torch.cat([feature, density], dim=1)
        feature = self.bn(feature)
        feature = self.relu(feature)
        feature = self.maxpool(feature)

        feature = self.layer1(feature)

        feature = self.layer2(feature)
        out_1st = None
        if self.fc1 is not None:
            out_1st = self.avgpool(feature)
            out_1st = torch.flatten(out_1st, 1)
            out_1st = self.fc1(out_1st)

        feature = self.layer3(feature)
        out_2nd = None
        if self.fc2 is not None:
            out_2nd = self.avgpool(feature)
            out_2nd = torch.flatten(out_2nd, 1)
            out_2nd = self.fc1(out_2nd)

        feature = self.layer4(feature)
        feature = self.avgpool(feature)
        feature = torch.flatten(feature, 1)
        out = self.fc(feature)

        return density_map, out, out_1st, out_2nd
