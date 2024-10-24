import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from utils.sam import SAM


class Detector(nn.Module):

    def __init__(self, architecture='EfficientNet'):
        super(Detector, self).__init__()
        self.net = EfficientNet.from_pretrained("efficientnet-b4", advprop=True, num_classes=2)

    def forward(self, x):
        x = self.net(x)
        return x

