import sys
from typing import Tuple, List, Dict, Any
import numpy as np
import torch
from torch import nn, Tensor

from .architecture_base import BaseModel
from .backbone_vgg import vgg16


class FcnPredictHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.in_channels = 64

        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(*[
            nn.Conv2d(3, 64, kernel_size=3, padding=100),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]))

        self._make_layer(2)
        self._make_layer(3)
        self._make_layer(3)
        self._make_layer(3, 512)
        self._make_layer(1, 4096, False)
        self._make_layer(1, 4096, False, kernel_size=7)

    def _make_layer(self, num_layers, out_channels=None, maxpool=True, **kwargs):
        layers = []
        if out_channels is None:
            out_channels = self.in_channels * 2
        else:
            out_channels = out_channels
        # if expansion:
        #     out_channels = 
        if not kwargs:
            kwargs = {'kernel_size': 3, 'padding': 1}

        for _ in range(num_layers):
            # out_channels = self.in_channels * 2
            layers += [
                nn.Conv2d(self.in_channels, out_channels, **kwargs),
                nn.ReLU()]
            self.in_channels = out_channels

        if maxpool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))        
        else:
            layers.append(nn.Dropout2d())

        self.layers.append(nn.Sequential(*layers))

    def forward(self, inputs):
        print(len(self.layers))
        outputs = []
        for layer in self.layers:
            inputs = layer(inputs)
            outputs.append(inputs)
        
        return outputs


class FcnModel(BaseModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # self.backbone = vgg16()
        self.head = FcnPredictHead(config)

    def forward(self, inputs):
        # outputs = self.backbone(inputs)
        outputs = self.head(inputs)
        return outputs