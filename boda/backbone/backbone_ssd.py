import math
from typing import Tuple, List, Dict

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class VGG(nn.Module):
    """
    This function is derived from torchvision VGG make_layers()
    https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
    https://github.com/dbolya/yolact/blob/master/backbone.py
    """
    def __init__(self, config, bn=True, num_classes=1000):
        super().__init__()

        self.bn = bn
        self.in_channels = 3
        self.channels = []
        self.layers = nn.ModuleList()

        for cfg in config:
            self._make_layers(cfg)

        self.avgpool = nn.AvgPool2d((7, 7))
        self.fc = None
      

    def forward(self, x):
        outputs = []
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        
        # x = self.avgpool(x)
        # x = self.classifier(x)

        return outputs

    def _make_layers(self, config):
        layers = []
        for v in config:
            kwargs = None
            if isinstance(v, tuple):
                kwargs = v[1]
                v = v[0]
                
            if v == 'M':
                if kwargs is None:
                    kwargs = {'kernel_size': 2, 'stride': 2}
                layers.append(
                    nn.MaxPool2d(**kwargs))
            else:
                if kwargs is None:
                    kwargs = {'kernel_size': 3, 'padding': 1}

                conv2d = nn.Conv2d(
                    in_channels=self.in_channels, 
                    out_channels=v, 
                    **kwargs)

                if self.bn:
                    layers += [
                        conv2d, nn.BatchNorm2d(v), nn.ReLU()]
                else:
                    layers += [conv2d, nn.ReLU()]

                self.in_channels = v
        
        self.channels.append(self.in_channels)
        self.layers.append(nn.Sequential(*layers))




def vgg(config: Dict = None):
    model = VGG(config)

    # if isinstance(config, dict):
    #     if config.backbone.pretrained:
    #         model.load_state_dict(torch.load(config.backbone.path))

    return model