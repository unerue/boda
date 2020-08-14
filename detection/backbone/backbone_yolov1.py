import math
from typing import List

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class TinyBlock(nn.Module):
    expansion = 2
    def __init__(self, in_channels, out_channels, kernel_size=3, pooling=True):
        super().__init__()
        self.pooling = pooling
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3,
            stride=1, 
            padding=3//2,
            bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU()
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.leaky(x)
        if self.pooling:
            x = self.mp(x)

        return x


class Darknet(nn.Module):
    def __init__(self, num_layers=9, block=TinyBlock):
        super().__init__()
        self.in_channels = 3
        self.layers = nn.ModuleList()
        self._make_layers(block, num_layers)
        self.fc = nn.Linear(256*3*3, 1470)

    def forward(self, x):
        outs = []
        for layer in self.layers:
            x = layer(x)
            outs.append(x)
        
        return outs

    def _make_layers(self, block, num_layers):
        out_channels = 16
        layers = []
        layers.append(block(self.in_channels, out_channels))
        
        pooling = True
        for i in range(3, num_layers):
            self.in_channels = out_channels
            out_channels = out_channels * block.expansion
            layers.append(block(self.in_channels, out_channels, pooling=pooling))
            if (i+1) == num_layers:
                pooling = False
            
        layers.append(block(out_channels, 256, pooling=pooling))
        self.layers.append(nn.Sequential(*layers))
        

def darknet9(pretrained=False, **kwargs):
    model = Darknet()
    if pretrained:
        model.load_state_dict(torch.load(pretrained))

    return model