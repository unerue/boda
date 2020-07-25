import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialPyramidPoolingNet(nn.Module):
    """Spatial Pyramid Pooling in Deep Convolutional Networks

    https://arxiv.org/abs/1406.4729

    A CNN model which adds spp layer so that we can input multi-size tensor
    """
    def __init__(self, opt, input_nc, ndf=64):
        super().__init__()
        self.output_num = [4, 2, 1]
        
        self.conv1 = nn.Conv2d(
            input_nc, ndf, kernel_size=4, stride=2, padding=1, bias=False)
        
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(ndf * 2)

        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ndf * 4)

        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ndf * 8)

        self.conv5 = nn.Conv2d(ndf * 8, 64, 4, 1, 0, bias=False)
        self.fc1 = nn.Linear(10752, 4096)
        self.fc2 = nn.Linear(4096, 1000)

    def forward(self, x):
        out = self.conv1(x)
        out = self.LReLU1(out)

        out = self.conv2(out)
        out = F.leaky_relu(self.bn1(out))

        out = self.conv3(out)
        out = F.leaky_relu(self.bn2(out))
        
        out = self.conv4(out)
        out = F.leaky_relu(self.bn3(out))
        
        out = self.conv5(out)
        out = self._spatial_pyramid_pooling(
            out, self.output_num)

        out = self.fc1(out)
        out = F.sigmoid(self.fc2(out))

        return out

from typing import List

