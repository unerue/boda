import math
from typing import Tuple, List, Dict

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from torchsummary import summary

from ..base import Backbone

class VGG(nn.Module):
    """
    This function is derived from torchvision VGG make_layers()
    https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
    https://github.com/dbolya/yolact/blob/master/backbone.py
    """
    def __init__(self, config, bn: bool = False, num_classes: int = 1000):
        super().__init__()
        self.bn = bn
        self.in_channels = 3
        self.channels = []
        self.layers = nn.ModuleList()

        for cfg in config:
            self._make_layer(cfg)

        self.avgpool = nn.AvgPool2d((7, 7))
        self.fc = None

    def forward(self, inputs):
        outputs = []
        for layer in self.layers:
            inputs = layer(inputs)
            outputs.append(inputs)

        return outputs

    def _make_layer(self, config):
        layers = []
        for v in config:
            kwargs = None
            if isinstance(v, tuple):
                kwargs = v[1]
                v = v[0]

            if v == 'M':
                if kwargs is None:
                    kwargs = {'kernel_size': 2, 'stride': 2}

                layers.append(nn.MaxPool2d(**kwargs))
            else:
                if kwargs is None:
                    kwargs = {'kernel_size': 3, 'padding': 1}

                conv2d = nn.Conv2d(
                    in_channels=self.in_channels, 
                    out_channels=v, 
                    **kwargs)

                if self.bn:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU()]
                else:
                    layers += [conv2d, nn.ReLU()]

                self.in_channels = v

        self.channels.append(self.in_channels)
        self.layers.append(nn.Sequential(*layers))


cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

structures = {
        'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'vgg16': [[64, 64],
              [ 'M', 128, 128],
              [ 'M', 256, 256, 256],
              [('M', {'kernel_size': 2, 'stride': 2, 'ceil_mode': True}), 512, 512, 512],
              [ 'M', 512, 512, 512],
              [('M',  {'kernel_size': 3, 'stride':  1, 'padding':  1}),
               (1024, {'kernel_size': 3, 'padding': 6, 'dilation': 6}),
               (1024, {'kernel_size': 1})]],
        'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
        'test': [
    [64, 64],
    ['M', 128, 128],
    ['M', 256, 256, 256],
    [('M', {'kernel_size': 2, 'stride': 2, 'ceil_mode': True}), 512, 512, 512],
    ['M', 512, 512, 512]],

    'test1': [
    [64, 64],
    [ 'M', 128, 128],
    [ 'M', 256, 256, 256],
    [('M', {'kernel_size': 2, 'stride': 2, 'ceil_mode': True}), 512, 512, 512],
    [ 'M', 512, 512, 512],
    [('M',  {'kernel_size': 3, 'stride':  1, 'padding':  1}),
     (1024, {'kernel_size': 3, 'padding': 6, 'dilation': 6}),
     (1024, {'kernel_size': 1})]
]
}

def vgg16(config: Dict = None):
    model = VGG(structures['test'])

    return model