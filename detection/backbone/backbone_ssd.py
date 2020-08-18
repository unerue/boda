import math
from typing import List

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class VGG(nn.Module):
    """
    This function is derived from torchvision VGG make_layers()
    https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
    """
    def __init__(self, config, batch_norm=True):
        super(VGG, self).__init__()
        self.layers = nn.ModuleList()
        self._make_layers(config)

    def forward(self, x):
        outputs = []
        for layer in self.layers:
            output = layer(x)
            outputs.append(output)

        return outputs

    def _make_layers(self, config, batch_norm=False):
        in_channels = 3
        for v in config:
            if v == 'M':
                self.layers.append([nn.MaxPool2d(kernel_size=2, stride=2)])
            elif v == 'C':
                self.layers.append([nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)])
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    self.layers.append([conv2d, nn.BatchNorm2d(v), nn.ReLU()])
                else:
                    self.layers.append([conv2d, nn.ReLU()])

                in_channels = v

        layers = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU(),
        )
        self.layers.append(layers)



