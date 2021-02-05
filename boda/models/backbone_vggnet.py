from collections import OrderedDict
import math
from typing import Tuple, List, Dict, Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.modules.batchnorm import BatchNorm2d
from ..base_architecture import Backbone


class VGG(nn.Module):
    """
    This function is derived from torchvision VGG make_layers()
    https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
    https://github.com/dbolya/yolact/blob/master/backbone.py
    """
    def __init__(
        self,
        structure,
        bn: bool = False,
        norm_layer: Optional[nn.Module] = nn.BatchNorm2d,
        num_classes: int = 1000
    ) -> None:
        super().__init__()
        self.bn = bn
        self.in_channels = 3
        self.channels = []
        self.layers = nn.ModuleList()

        for layer in structure:
            self._make_layer(layer)

    def forward(self, inputs):
        outputs = []
        for layer in self.layers:
            inputs = layer(inputs)
            outputs.append(inputs)

        return outputs

    def _make_layer(self, config):
        # _layers = []
        _layers = OrderedDict()
        i = 0
        for v in config:
            kwargs = None
            if isinstance(v, tuple):
                kwargs = v[1]
                v = v[0]

            if v == 'M':
                if kwargs is None:
                    kwargs = {'kernel_size': 2, 'stride': 2}

                # _layers.append(nn.MaxPool2d(**kwargs))
                # _layers.update({'maxpool': nn.MaxPool2d(**kwargs)})
                _layers.update({f'maxpool{i}': nn.MaxPool2d(**kwargs)})
            else:
                if kwargs is None:
                    kwargs = {'kernel_size': 3, 'padding': 1}

                conv2d = nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=v,
                    **kwargs
                )

                if self.bn:
                    # _layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU()]
                    # _layers.update({'conv': conv2d, 'bn': nn.BatchNorm2d(v), 'relu': nn.ReLU()})
                    _layers.update({f'{i}': conv2d, f'bn{i}': nn.BatchNorm2d(v), f'relu{i}': nn.ReLU()})
                else:
                    # _layers += [conv2d, nn.ReLU()]
                    # _layers.update({'conv': conv2d, 'relu': nn.ReLU()})
                    _layers.update({f'{i}': conv2d, f'relu{i}': nn.ReLU()})

                self.in_channels = v
                i += 1

        self.channels.append(self.in_channels)
        self.layers.append(nn.Sequential(_layers))


structures = {
    'vgg16': [
        [64, 64],
        ['M', 128, 128],
        ['M', 256, 256, 256],
        [('M', {'kernel_size': 2, 'stride': 2, 'ceil_mode': True}), 512, 512, 512],
        ['M', 512, 512, 512]]}


def vgg16(config: Dict = None):
    model = VGG(structures['vgg16'])

    return model