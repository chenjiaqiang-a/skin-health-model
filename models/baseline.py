import torch.nn as nn
import torchvision.models as models


class ResNet50Baseline(nn.Module):
    def __init__(self, output_dim=8):
        super(ResNet50Baseline, self).__init__()

        self.resnet = models.resnet50(pretrained=True)

        self.resnet.fc = nn.Linear(2048, output_dim)

    def forward(self, x):
        x = self.resnet(x)
        return x
